use std::collections::BTreeSet;

use charabia::normalizer::NormalizedTokenIter;
use charabia::{SeparatorKind, TokenKind, Tokenizer};

use super::compute_derivations::partially_initialized_term_from_word;
use super::{LocatedQueryTerm, ZeroTypoTerm};
use crate::search::new::query_graph::QueryGraph;
use crate::search::new::query_term::{Lazy, Phrase, QueryTerm};
use crate::search::new::Word;
use crate::{Result, SearchContext, MAX_WORD_LENGTH};

#[derive(Clone)]
/// Extraction of the content of a query.
pub struct ExtractedTokens {
    /// The terms to search for in the database.
    pub query_terms: Vec<LocatedQueryTerm>,
    /// The graph associated with these query terms
    pub graph: QueryGraph,
    /// The words that must not appear in the results.
    pub negative_words: Vec<Word>,
    /// The phrases that must not appear in the results.
    pub negative_phrases: Vec<LocatedQueryTerm>,
}

fn located_word_term(
    ctx: &mut SearchContext<'_>,
    tokenizer: &Tokenizer<'_>,
    token: &charabia::Token<'_>,
    position: u16,
    max_typo: u8,
    is_prefix: bool,
) -> Result<LocatedQueryTerm> {
    let word = token.lemma();

    #[cfg(feature = "japanese")]
    let japanese_variants = crate::japanese::search_variants(token);
    #[cfg(feature = "japanese")]
    let search_alternates = japanese_variants.iter_unique().cloned().collect::<Vec<_>>();
    #[cfg(not(feature = "japanese"))]
    let search_alternates = Vec::new();

    let term = partially_initialized_term_from_word(
        ctx,
        tokenizer,
        word,
        &search_alternates,
        max_typo,
        is_prefix,
        false,
    )?;

    Ok(LocatedQueryTerm {
        value: ctx.term_interner.push(term),
        positions: position..=position,
        #[cfg(feature = "japanese")]
        japanese_variants: japanese_variants.map(|variant| ctx.word_interner.insert(variant)),
    })
}

/// Convert the tokenised search query into a list of located query terms.
#[tracing::instrument(level = "trace", skip_all, target = "search::query")]
pub fn located_query_terms_from_tokens(
    ctx: &mut SearchContext<'_>,
    tokenizer: &Tokenizer<'_>,
    query: NormalizedTokenIter<'_, '_, '_, '_>,
    words_limit: Option<usize>,
    japanese_search_enabled: bool,
) -> Result<ExtractedTokens> {
    #[cfg(not(feature = "japanese"))]
    let _ = japanese_search_enabled;
    let nbr_typos = number_of_typos_allowed(ctx)?;
    let allow_prefix_search = ctx.is_prefix_search_allowed();

    let mut query_terms = Vec::new();

    let mut negative_phrase = false;
    let mut phrase: Option<PhraseBuilder> = None;
    let mut encountered_whitespace = true;
    let mut negative_next_token = false;
    let mut negative_words = Vec::new();
    let mut negative_phrases = Vec::new();

    let parts_limit = words_limit.unwrap_or(usize::MAX);

    // start with the last position as we will wrap around to position 0 at the beginning of the loop below.
    let mut position = u16::MAX;

    let mut peekable = query.take(super::limits::MAX_TOKEN_COUNT).peekable();
    while let Some(token) = peekable.next() {
        if token.lemma().is_empty() {
            continue;
        }

        // early return if word limit is exceeded
        if query_terms.len() >= parts_limit {
            let (graph, query_terms) =
                QueryGraph::from_query(ctx, tokenizer, &query_terms, japanese_search_enabled)?;

            return Ok(ExtractedTokens { query_terms, graph, negative_words, negative_phrases });
        }

        match token.kind {
            TokenKind::Word | TokenKind::StopWord => {
                // On first loop, goes from u16::MAX to 0, then normal increment.
                position = position.wrapping_add(1);

                // 1. if the word is quoted we push it in a phrase-buffer waiting for the ending quote,
                // 2. if the word is not the last token of the query and is not a stop_word we push it as a non-prefix word,
                // 3. if the word is the last token of the query we push it as a prefix word.
                if let Some(phrase) = &mut phrase {
                    phrase.push_word(ctx, &token, position)
                } else if negative_next_token {
                    let word = token.lemma().to_string();
                    #[cfg(feature = "japanese")]
                    let surface_is_indexed = ctx.index.contains_word(ctx.txn, &word)?;
                    negative_words.push(Word::Original(ctx.word_interner.insert(word)));

                    #[cfg(feature = "japanese")]
                    if japanese_search_enabled && !surface_is_indexed {
                        for alternate in crate::japanese::search_variants(&token).iter_unique() {
                            if alternate.len() <= MAX_WORD_LENGTH
                                && ctx.index.contains_word(ctx.txn, alternate)?
                            {
                                negative_words.push(Word::Variant(
                                    ctx.word_interner.insert(alternate.clone()),
                                ));
                            }
                        }
                    }
                    negative_next_token = false;
                } else if peekable.peek().is_some() {
                    match token.kind {
                        TokenKind::Word => {
                            let word = token.lemma();
                            query_terms.push(located_word_term(
                                ctx,
                                tokenizer,
                                &token,
                                position,
                                nbr_typos(word),
                                false,
                            )?);
                        }
                        TokenKind::StopWord | TokenKind::Separator(_) | TokenKind::Unknown => (),
                    }
                } else {
                    let word = token.lemma();
                    query_terms.push(located_word_term(
                        ctx,
                        tokenizer,
                        &token,
                        position,
                        nbr_typos(word),
                        allow_prefix_search,
                    )?);
                }
            }
            TokenKind::Separator(separator_kind) => {
                // add penalty for hard separators
                if let SeparatorKind::Hard = separator_kind {
                    position = position.wrapping_add(7);
                }

                phrase = 'phrase: {
                    let phrase = phrase.take();

                    // If we have a hard separator inside a phrase, we immediately start a new phrase
                    let phrase = if separator_kind == SeparatorKind::Hard {
                        if let Some(phrase) = phrase {
                            if let Some(located_query_term) = phrase.build(ctx)? {
                                // as we are evaluating a negative operator we put the phrase
                                // in the negative one *but* we don't reset the negative operator
                                // as we are immediately starting a new negative phrase.
                                if negative_phrase {
                                    negative_phrases.push(located_query_term);
                                } else {
                                    query_terms.push(located_query_term);
                                }
                            }
                            Some(PhraseBuilder::empty(japanese_search_enabled))
                        } else {
                            None
                        }
                    } else {
                        phrase
                    };

                    // We close and start a new phrase depending on the number of double quotes
                    let mut quote_count = token.lemma().chars().filter(|&s| s == '"').count();
                    if quote_count == 0 {
                        break 'phrase phrase;
                    }

                    // Consume the closing quote and the phrase
                    if let Some(phrase) = phrase {
                        // Per the check above, quote_count > 0
                        quote_count -= 1;
                        if let Some(located_query_term) = phrase.build(ctx)? {
                            // we were evaluating a negative operator so we
                            // put the phrase in the negative phrases
                            if negative_phrase {
                                negative_phrases.push(located_query_term);
                                negative_phrase = false;
                            } else {
                                query_terms.push(located_query_term);
                            }
                        }
                    }

                    // Start new phrase if the token ends with an opening quote
                    if quote_count % 2 == 1 {
                        negative_phrase = negative_next_token;
                        Some(PhraseBuilder::empty(japanese_search_enabled))
                    } else {
                        None
                    }
                };

                negative_next_token =
                    phrase.is_none() && token.lemma() == "-" && encountered_whitespace;
            }
            _ => (),
        }

        encountered_whitespace =
            token.lemma().chars().last().filter(|c| c.is_whitespace()).is_some();
    }

    // If a quote is never closed, we consider all of the end of the query as a phrase.
    if let Some(phrase) = phrase.take() {
        if let Some(located_query_term) = phrase.build(ctx)? {
            // put the phrase in the negative set if we are evaluating a negative operator.
            if negative_phrase {
                negative_phrases.push(located_query_term);
            } else {
                query_terms.push(located_query_term);
            }
        }
    }

    let (graph, query_terms) =
        QueryGraph::from_query(ctx, tokenizer, &query_terms, japanese_search_enabled)?;

    Ok(ExtractedTokens { query_terms, graph, negative_words, negative_phrases })
}

pub fn number_of_typos_allowed<'ctx>(
    ctx: &SearchContext<'ctx>,
) -> Result<impl Fn(&str) -> u8 + 'ctx> {
    let authorize_typos = ctx.index.authorize_typos(ctx.txn)?;
    let min_len_one_typo = ctx.index.min_word_len_one_typo(ctx.txn)?;
    let min_len_two_typos = ctx.index.min_word_len_two_typos(ctx.txn)?;

    let exact_words = ctx.index.exact_words(ctx.txn)?;

    Ok(Box::new(move |word: &str| {
        if !authorize_typos
            || word.chars().count() < min_len_one_typo as usize
            || exact_words.as_ref().is_some_and(|fst| fst.contains(word))
        {
            0
        } else if word.chars().count() < min_len_two_typos as usize {
            1
        } else {
            2
        }
    }))
}

pub fn make_ngram(
    ctx: &mut SearchContext<'_>,
    tokenizer: &Tokenizer<'_>,
    terms: &[LocatedQueryTerm],
    number_of_typos_allowed: &impl Fn(&str) -> u8,
) -> Result<Option<LocatedQueryTerm>> {
    assert!(!terms.is_empty());
    for t in terms {
        if ctx.term_interner.get(t.value).zero_typo.phrase.is_some() {
            return Ok(None);
        }
    }
    for ts in terms.windows(2) {
        let [t1, t2] = ts else { panic!() };
        if *t1.positions.end() != t2.positions.start() - 1 {
            return Ok(None);
        }
    }
    let mut words_interned = vec![];
    for term in terms {
        if let Some(original_term_word) = term.value.original_single_word(ctx) {
            words_interned.push(original_term_word);
        } else {
            return Ok(None);
        }
    }
    let words =
        words_interned.iter().map(|&i| ctx.word_interner.get(i).to_owned()).collect::<Vec<_>>();

    let start = *terms.first().as_ref().unwrap().positions.start();
    let end = *terms.last().as_ref().unwrap().positions.end();
    let is_prefix = ctx.term_interner.get(terms.last().as_ref().unwrap().value).is_prefix;
    let ngram_str = words.join("");
    if ngram_str.len() > MAX_WORD_LENGTH {
        return Ok(None);
    }
    let ngram_str_interned = ctx.word_interner.insert(ngram_str.clone());

    let max_nbr_typos =
        number_of_typos_allowed(ngram_str.as_str()).saturating_sub(terms.len() as u8 - 1);

    #[cfg(feature = "japanese")]
    let japanese_variants = super::japanese::concatenate_variants(ctx, terms);

    let mut term = partially_initialized_term_from_word(
        ctx,
        tokenizer,
        &ngram_str,
        &[],
        max_nbr_typos,
        is_prefix,
        true,
    )?;

    // Now add the synonyms
    if let Some(synonyms) = ctx.index.synonyms.get(ctx.txn, &words)? {
        for synonym in synonyms.synonyms(tokenizer) {
            let words =
                synonym.into_iter().map(|w| Some(ctx.word_interner.insert(w.to_owned()))).collect();
            let interned = ctx.phrase_interner.insert(Phrase { words });
            term.zero_typo.synonyms.insert(interned);
        }
    }

    let term = QueryTerm {
        original: ngram_str_interned,
        ngram_words: Some(words_interned),
        is_prefix,
        max_levenshtein_distance: max_nbr_typos,
        ranking_span_len: terms.len() as u8,
        zero_typo: term.zero_typo,
        one_typo: Lazy::Uninit,
        two_typo: Lazy::Uninit,
    };

    let term = LocatedQueryTerm {
        value: ctx.term_interner.push(term),
        positions: start..=end,
        #[cfg(feature = "japanese")]
        japanese_variants: japanese_variants.map(|variant| ctx.word_interner.insert(variant)),
    };

    Ok(Some(term))
}

struct PhraseBuilder {
    words: Vec<Option<crate::search::new::Interned<String>>>,
    #[cfg(feature = "japanese")]
    japanese_variants: Vec<crate::japanese::SearchVariants<crate::search::new::Interned<String>>>,
    #[cfg(feature = "japanese")]
    japanese_search_enabled: bool,
    start: u16,
    end: u16,
}

impl PhraseBuilder {
    fn empty(japanese_search_enabled: bool) -> Self {
        #[cfg(not(feature = "japanese"))]
        let _ = japanese_search_enabled;

        Self {
            words: Default::default(),
            #[cfg(feature = "japanese")]
            japanese_variants: Default::default(),
            #[cfg(feature = "japanese")]
            japanese_search_enabled,
            start: u16::MAX,
            end: u16::MAX,
        }
    }

    fn is_empty(&self) -> bool {
        self.words.is_empty() || self.words.iter().all(Option::is_none)
    }

    // precondition: token has kind Word or StopWord
    fn push_word(
        &mut self,
        ctx: &mut SearchContext<'_>,
        token: &charabia::Token<'_>,
        position: u16,
    ) {
        if self.is_empty() {
            self.start = position;
        }
        self.end = position;
        if let TokenKind::StopWord = token.kind {
            self.words.push(None);
            #[cfg(feature = "japanese")]
            self.japanese_variants.push(Default::default());
        } else {
            // token has kind Word
            let word = ctx.word_interner.insert(token.lemma().to_string());
            self.words.push(Some(word));
            #[cfg(feature = "japanese")]
            self.japanese_variants.push(
                crate::japanese::search_variants(token)
                    .map(|variant| ctx.word_interner.insert(variant)),
            );
        }
    }

    fn build(self, ctx: &mut SearchContext<'_>) -> Result<Option<LocatedQueryTerm>> {
        if self.is_empty() {
            return Ok(None);
        }

        let phrase = ctx.phrase_interner.insert(Phrase { words: self.words });

        #[cfg(feature = "japanese")]
        let alternate_phrases = if self.japanese_search_enabled {
            super::japanese::alternate_phrases(ctx, phrase, &self.japanese_variants)?
        } else {
            BTreeSet::new()
        };
        #[cfg(not(feature = "japanese"))]
        let alternate_phrases = BTreeSet::new();

        let phrase_desc = phrase.description(ctx);

        Ok(Some(LocatedQueryTerm {
            value: ctx.term_interner.push(QueryTerm {
                original: ctx.word_interner.insert(phrase_desc),
                ngram_words: None,
                max_levenshtein_distance: 0,
                is_prefix: false,
                ranking_span_len: 1,
                zero_typo: ZeroTypoTerm {
                    phrase: Some(phrase),
                    exact: None,
                    alternates: BTreeSet::default(),
                    alternate_phrases,
                    prefix_of: BTreeSet::default(),
                    synonyms: BTreeSet::default(),
                    use_prefix_db: None,
                },
                one_typo: Lazy::Uninit,
                two_typo: Lazy::Uninit,
            }),
            positions: self.start..=self.end,
            #[cfg(feature = "japanese")]
            japanese_variants: Default::default(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use charabia::TokenizerBuilder;

    use super::*;
    use crate::index::tests::TempIndex;

    fn temp_index_with_documents() -> TempIndex {
        let temp_index = TempIndex::new();
        temp_index
            .add_documents(documents!([
                { "id": 1, "name": "split this world westfali westfalia the Ŵôřlḑôle" },
                { "id": 2, "name": "Westfália" },
                { "id": 3, "name": "Ŵôřlḑôle" },
            ]))
            .unwrap();
        temp_index
    }

    #[test]
    fn start_with_hard_separator() -> Result<()> {
        let mut builder = TokenizerBuilder::default();
        let tokenizer = builder.build();
        let tokens = tokenizer.tokenize(".");
        let index = temp_index_with_documents();
        let rtxn = index.read_txn()?;
        let fields_ids_map = index.fields_ids_map(&rtxn)?;
        let mut ctx = SearchContext::new(
            &index,
            &rtxn,
            &fields_ids_map,
            "test",
            time::OffsetDateTime::now_utc(),
        )?;
        // panics with `attempt to add with overflow` before <https://github.com/meilisearch/meilisearch/issues/3785>
        let ExtractedTokens { query_terms, .. } =
            located_query_terms_from_tokens(&mut ctx, &tokenizer, tokens, None)?;
        assert!(query_terms.is_empty());

        Ok(())
    }

    #[test]
    fn test_unicode_typo_tolerance_fixed() -> Result<()> {
        let temp_index = temp_index_with_documents();
        let rtxn = temp_index.read_txn()?;
        let fields_ids_map = temp_index.fields_ids_map(&rtxn)?;
        let ctx = SearchContext::new(
            &temp_index,
            &rtxn,
            &fields_ids_map,
            "test",
            time::OffsetDateTime::now_utc(),
        )?;

        let nbr_typos = number_of_typos_allowed(&ctx)?;

        // ASCII word "doggy" (5 chars, 5 bytes)
        let ascii_word = "doggy";
        let ascii_typos = nbr_typos(ascii_word);

        // Cyrillic word "собак" (5 chars, 10 bytes)
        let cyrillic_word = "собак";
        let cyrillic_typos = nbr_typos(cyrillic_word);

        // Both words have 5 characters, so they should have the same typo tolerance
        assert_eq!(
            ascii_typos, cyrillic_typos,
            "Words with same character count should get same typo tolerance"
        );

        // With default settings (oneTypo=5, twoTypos=9), 5-char words should get 1 typo
        assert_eq!(ascii_typos, 1, "5-character word should get 1 typo tolerance");
        assert_eq!(cyrillic_typos, 1, "5-character word should get 1 typo tolerance");

        Ok(())
    }

    #[test]
    fn test_various_unicode_scripts() -> Result<()> {
        let temp_index = temp_index_with_documents();
        let rtxn = temp_index.read_txn()?;
        let fields_ids_map = temp_index.fields_ids_map(&rtxn)?;
        let ctx = SearchContext::new(
            &temp_index,
            &rtxn,
            &fields_ids_map,
            "test",
            time::OffsetDateTime::now_utc(),
        )?;

        let nbr_typos = number_of_typos_allowed(&ctx)?;

        // Let's use 5-character words for consistent testing
        let five_char_words = vec![
            ("doggy", "ASCII"),    // 5 chars, 5 bytes
            ("café!", "Accented"), // 5 chars, 7 bytes
            ("собак", "Cyrillic"), // 5 chars, 10 bytes
        ];

        let expected_typos = 1; // With default settings, 5-char words get 1 typo

        for (word, script) in five_char_words {
            let typos = nbr_typos(word);
            assert_eq!(
                typos, expected_typos,
                "{} word '{}' should get {} typo(s)",
                script, word, expected_typos
            );
        }

        Ok(())
    }
}
