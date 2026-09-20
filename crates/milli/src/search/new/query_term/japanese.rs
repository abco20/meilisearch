//! Query-time Japanese resegmentation.
//!
//! Charabia and the indexed document can choose different Japanese token boundaries. This module
//! bridges that mismatch against the indexed-word FST without persisting multi-token span terms.
//! Semantic synonyms remain handled by the normal synonym path.

use std::collections::BTreeSet;

use super::{Lazy, LocatedQueryTerm, Phrase, QueryTerm, ZeroTypoTerm};
use crate::japanese::SearchVariants;
use crate::search::new::interner::Interned;
use crate::search::new::SearchContext;
use crate::{Result, MAX_WORD_LENGTH};

const MAX_RESEGMENTATION_TERMS: usize = 16;
const MAX_RESEGMENTATION_PATHS: usize = 5;
const MAX_RESEGMENTATION_NODES: usize = 3;

#[derive(Clone)]
pub(in crate::search::new) struct ResegmentationCandidate {
    pub(in crate::search::new) attach_at: usize,
    pub(in crate::search::new) start_idx: usize,
    pub(in crate::search::new) end_idx: usize,
    pub(in crate::search::new) term: LocatedQueryTerm,
}

pub(super) fn concatenate_variants(
    ctx: &SearchContext<'_>,
    terms: &[LocatedQueryTerm],
) -> SearchVariants<String> {
    let mut variants = SearchVariants {
        normalized: Some(String::new()),
        phonetic: Some(String::new()),
        romaji: Some(String::new()),
    };

    for term in terms {
        match (&mut variants.normalized, term.japanese_variants.normalized) {
            (Some(value), Some(part)) => value.push_str(ctx.word_interner.get(part)),
            (value, None) => *value = None,
            (None, Some(_)) => {}
        }
        match (&mut variants.phonetic, term.japanese_variants.phonetic) {
            (Some(value), Some(part)) => value.push_str(ctx.word_interner.get(part)),
            (value, None) => *value = None,
            (None, Some(_)) => {}
        }
        match (&mut variants.romaji, term.japanese_variants.romaji) {
            (Some(value), Some(part)) => value.push_str(ctx.word_interner.get(part)),
            (value, None) => *value = None,
            (None, Some(_)) => {}
        }
    }

    if let Some(normalized) = variants.normalized.as_mut() {
        *normalized = crate::japanese::normalize_compact_reading(normalized);
    }
    if let Some(phonetic) = variants.phonetic.as_mut() {
        *phonetic = crate::japanese::normalize_compact_reading(phonetic);
    }

    variants
}

fn resegment_indexed_variants(
    ctx: &mut SearchContext<'_>,
    value: &str,
) -> Result<Vec<Vec<String>>> {
    if value.is_empty() {
        return Ok(Vec::new());
    }

    let mut boundaries = value.char_indices().map(|(index, _)| index).collect::<Vec<_>>();
    boundaries.push(value.len());
    if boundaries.len() <= 1 {
        return Ok(Vec::new());
    }

    let words_fst = ctx.get_words_fst()?;
    let chars = boundaries.len() - 1;
    let mut best = vec![Vec::<Vec<String>>::new(); chars + 1];
    best[chars].push(Vec::new());

    for start_idx in (0..chars).rev() {
        let mut candidates = Vec::new();
        for end_idx in (start_idx + 1..=chars).rev() {
            let start = boundaries[start_idx];
            let end = boundaries[end_idx];
            if end - start > MAX_WORD_LENGTH || best[end_idx].is_empty() {
                continue;
            }

            let candidate = &value[start..end];
            if !words_fst.contains(candidate) {
                continue;
            }

            for suffix in &best[end_idx] {
                let mut path = Vec::with_capacity(1 + suffix.len());
                path.push(candidate.to_owned());
                path.extend(suffix.iter().cloned());
                candidates.push(path);
            }
        }

        candidates.sort_by(|left, right| {
            left.len()
                .cmp(&right.len())
                .then_with(|| right.first().map(String::len).cmp(&left.first().map(String::len)))
                .then_with(|| left.cmp(right))
        });
        candidates.dedup();
        candidates.truncate(MAX_RESEGMENTATION_PATHS);
        best[start_idx] = candidates;
    }

    Ok(best.into_iter().next().unwrap_or_default())
}

fn segmentation_is_already_termwise(
    ctx: &SearchContext<'_>,
    terms: &[LocatedQueryTerm],
    segments: &[String],
) -> bool {
    if terms.len() != segments.len() {
        return false;
    }

    terms.iter().zip(segments).all(|(term, segment)| {
        let original_matches = term
            .value
            .original_single_word(ctx)
            .is_some_and(|word| ctx.word_interner.get(word) == segment);

        original_matches
            || term
                .japanese_variants
                .iter_unique()
                .any(|word| ctx.word_interner.get(*word) == segment)
    })
}

fn term_has_indexed_representation(ctx: &SearchContext<'_>, term: &LocatedQueryTerm) -> bool {
    let query_term = ctx.term_interner.get(term.value);
    query_term.zero_typo.exact.is_some() || !query_term.zero_typo.alternates.is_empty()
}

fn original_terms_have_phrase_match(
    ctx: &mut SearchContext<'_>,
    terms: &[LocatedQueryTerm],
) -> Result<bool> {
    let mut words = Vec::with_capacity(terms.len());
    for term in terms {
        let Some(word) = term.value.original_single_word(ctx) else {
            return Ok(false);
        };
        words.push(Some(word));
    }
    let phrase = ctx.phrase_interner.insert(Phrase { words });
    Ok(!ctx.get_phrase_docids(phrase)?.is_empty())
}

fn make_resegmentation(
    ctx: &mut SearchContext<'_>,
    terms: &[LocatedQueryTerm],
) -> Result<Option<LocatedQueryTerm>> {
    if terms.is_empty() {
        return Ok(None);
    }

    for term in terms {
        if ctx.term_interner.get(term.value).zero_typo.phrase.is_some() {
            return Ok(None);
        }
    }
    for pair in terms.windows(2) {
        let [left, right] = pair else { unreachable!() };
        if *left.positions.end() != right.positions.start() - 1 {
            return Ok(None);
        }
    }

    let japanese_variants = concatenate_variants(ctx, terms);
    let mut original_words = Vec::with_capacity(terms.len());
    for term in terms {
        let Some(original_word) = term.value.original_single_word(ctx) else {
            return Ok(None);
        };
        original_words.push(original_word);
    }

    let mut alternates = BTreeSet::new();
    let mut alternate_phrases = BTreeSet::new();

    for compact in japanese_variants.iter_unique() {
        if compact.is_empty() || compact.len() > MAX_WORD_LENGTH {
            continue;
        }

        for segments in resegment_indexed_variants(ctx, compact)? {
            if segmentation_is_already_termwise(ctx, terms, &segments) {
                continue;
            }

            if segments.len() == 1 {
                alternates.insert(ctx.word_interner.insert(segments.into_iter().next().unwrap()));
                break;
            }

            let words =
                segments.into_iter().map(|word| Some(ctx.word_interner.insert(word))).collect();
            let phrase = ctx.phrase_interner.insert(Phrase { words });
            if !ctx.get_phrase_docids(phrase)?.is_empty() {
                alternate_phrases.insert(phrase);
                break;
            }
        }
    }

    if alternates.is_empty() && alternate_phrases.is_empty() {
        return Ok(None);
    }

    let start = *terms.first().unwrap().positions.start();
    let end = *terms.last().unwrap().positions.end();
    let original =
        original_words.iter().map(|word| ctx.word_interner.get(*word).as_str()).collect::<String>();

    let term = QueryTerm {
        original: ctx.word_interner.insert(original),
        ngram_words: Some(original_words),
        max_levenshtein_distance: 0,
        is_prefix: false,
        ranking_span_len: 1,
        zero_typo: ZeroTypoTerm { alternates, alternate_phrases, ..ZeroTypoTerm::default() },
        one_typo: Lazy::Init(Default::default()),
        two_typo: Lazy::Init(Default::default()),
    };

    Ok(Some(LocatedQueryTerm {
        value: ctx.term_interner.push(term),
        positions: start..=end,
        japanese_variants: japanese_variants.map(|variant| ctx.word_interner.insert(variant)),
    }))
}

pub(in crate::search::new) fn resegmentation_candidates(
    ctx: &mut SearchContext<'_>,
    terms: &[LocatedQueryTerm],
) -> Result<Vec<ResegmentationCandidate>> {
    let mut output = Vec::new();
    let mut run_start = 0;

    while run_start < terms.len() {
        if terms[run_start].japanese_variants.normalized.is_none() {
            run_start += 1;
            continue;
        }

        let mut run_end = run_start;
        while run_end + 1 < terms.len()
            && terms[run_end + 1].japanese_variants.normalized.is_some()
            && *terms[run_end].positions.end() + 1 == *terms[run_end + 1].positions.start()
        {
            run_end += 1;
        }

        collect_run_candidates(ctx, terms, run_start, run_end, &mut output)?;
        run_start = run_end + 1;
    }

    Ok(output)
}

fn collect_run_candidates(
    ctx: &mut SearchContext<'_>,
    terms: &[LocatedQueryTerm],
    run_start: usize,
    run_end: usize,
    output: &mut Vec<ResegmentationCandidate>,
) -> Result<()> {
    let run_len = run_end - run_start + 1;
    let mut first_missing = None;
    let mut last_missing = None;

    for index in run_start..=run_end {
        if !term_has_indexed_representation(ctx, &terms[index]) {
            first_missing.get_or_insert(index);
            last_missing = Some(index);
        }
    }

    if let (Some(first_missing), Some(last_missing)) = (first_missing, last_missing) {
        let core_len = last_missing - first_missing + 1;
        if core_len > MAX_RESEGMENTATION_TERMS {
            return Ok(());
        }

        let max_extra = MAX_RESEGMENTATION_TERMS - core_len;
        let mut candidate_count = 0;
        'find_candidates: for extra in 0..=max_extra {
            for left_extra in 0..=extra {
                let right_extra = extra - left_extra;
                if left_extra > first_missing - run_start || right_extra > run_end - last_missing {
                    continue;
                }

                let start_idx = first_missing - left_extra;
                let end_idx = last_missing + right_extra;
                if let Some(term) = make_resegmentation(ctx, &terms[start_idx..=end_idx])? {
                    output.push(ResegmentationCandidate {
                        attach_at: run_end,
                        start_idx,
                        end_idx,
                        term,
                    });
                    candidate_count += 1;
                    if candidate_count >= MAX_RESEGMENTATION_NODES {
                        break 'find_candidates;
                    }
                }
            }
        }
    } else if run_len > 1
        && run_len <= MAX_RESEGMENTATION_TERMS
        && !original_terms_have_phrase_match(ctx, &terms[run_start..=run_end])?
    {
        if let Some(term) = make_resegmentation(ctx, &terms[run_start..=run_end])? {
            output.push(ResegmentationCandidate {
                attach_at: run_end,
                start_idx: run_start,
                end_idx: run_end,
                term,
            });
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum PhraseVariantKind {
    Normalized,
    Phonetic,
    Romaji,
}

pub(super) fn alternate_phrases(
    ctx: &mut SearchContext<'_>,
    original: Interned<Phrase>,
    variants: &[SearchVariants<Interned<String>>],
) -> Result<BTreeSet<Interned<Phrase>>> {
    if !ctx.get_phrase_docids(original)?.is_empty() {
        return Ok(BTreeSet::new());
    }

    let mut alternates = BTreeSet::new();
    for kind in
        [PhraseVariantKind::Normalized, PhraseVariantKind::Phonetic, PhraseVariantKind::Romaji]
    {
        if let Some(phrase) = phrase_variant(ctx, variants, kind)? {
            alternates.insert(phrase);
        }
    }
    Ok(alternates)
}

fn phrase_variant(
    ctx: &mut SearchContext<'_>,
    variants: &[SearchVariants<Interned<String>>],
    kind: PhraseVariantKind,
) -> Result<Option<Interned<Phrase>>> {
    let mut output = Vec::with_capacity(variants.len());
    let mut changed = false;
    let mut index = 0;

    while index < variants.len() {
        let has_variant = |variants: &SearchVariants<_>| {
            variants.normalized.is_some()
                || variants.phonetic.is_some()
                || variants.romaji.is_some()
        };

        if !has_variant(&variants[index]) {
            output.push(None);
            index += 1;
            continue;
        }

        let run_start = index;
        let mut run_end = index;
        while run_end + 1 < variants.len() && has_variant(&variants[run_end + 1]) {
            run_end += 1;
        }

        let mut compact = String::new();
        let mut complete = true;
        for token_variants in &variants[run_start..=run_end] {
            let part = match kind {
                PhraseVariantKind::Normalized => token_variants.normalized,
                PhraseVariantKind::Phonetic => token_variants.phonetic,
                PhraseVariantKind::Romaji => token_variants.romaji,
            };
            let Some(part) = part else {
                complete = false;
                break;
            };
            compact.push_str(ctx.word_interner.get(part));
        }

        if matches!(kind, PhraseVariantKind::Normalized | PhraseVariantKind::Phonetic) {
            compact = crate::japanese::normalize_compact_reading(&compact);
        }

        let segmentations = if complete && !compact.is_empty() && compact.len() <= MAX_WORD_LENGTH {
            resegment_indexed_variants(ctx, &compact)?
        } else {
            Vec::new()
        };

        let Some(segments) = segmentations.into_iter().next() else {
            return Ok(None);
        };
        changed = true;
        output.extend(segments.into_iter().map(|word| Some(ctx.word_interner.insert(word))));

        index = run_end + 1;
    }

    if !changed {
        return Ok(None);
    }

    Ok(Some(ctx.phrase_interner.insert(Phrase { words: output })))
}
