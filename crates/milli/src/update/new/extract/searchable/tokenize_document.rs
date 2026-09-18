use std::collections::HashMap;

#[cfg(feature = "japanese")]
use charabia::segmenter::JapaneseSegmenter;
#[cfg(feature = "japanese")]
use charabia::{Language, TokenMetadata};
use charabia::{SeparatorKind, Token, TokenKind, Tokenizer, TokenizerBuilder};
use serde_json::Value;

use crate::attribute_patterns::PatternMatch;
use crate::update::new::document::Document;
use crate::update::new::extract::perm_json_p::{
    seek_leaf_values_in_array, seek_leaf_values_in_object, Depth,
};
use crate::{FieldId, InternalError, LocalizedAttributesRule, Result, MAX_WORD_LENGTH};

// todo: should be crate::proximity::MAX_DISTANCE but it has been forgotten
const MAX_DISTANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SearchTermKind {
    Original,
    #[cfg(feature = "japanese")]
    Alternate,
}

impl SearchTermKind {
    pub(super) fn is_original(self) -> bool {
        matches!(self, Self::Original)
    }
}

#[derive(Clone, Copy)]
pub struct DocumentTokenizer<'a> {
    pub tokenizer: &'a Tokenizer<'a>,
    pub localized_attributes_rules: &'a [LocalizedAttributesRule],
    pub max_positions_per_attributes: u32,
}

impl DocumentTokenizer<'_> {
    pub fn tokenize_document<'doc>(
        &self,
        document: impl Document<'doc>,
        should_tokenize: &mut impl FnMut(&str) -> Result<(FieldId, PatternMatch)>,
        token_fn: &mut impl FnMut(&str, FieldId, u16, &str, SearchTermKind) -> Result<()>,
    ) -> Result<()> {
        let mut field_position = HashMap::new();
        for entry in document.iter_top_level_fields() {
            let (field_name, value) = entry?;

            if let (_, PatternMatch::NoMatch) = should_tokenize(field_name)? {
                continue;
            }

            let mut tokenize_field = |field_name: &str, _depth, value: &Value| {
                let (fid, pattern_match) = should_tokenize(field_name)?;
                if pattern_match == PatternMatch::Match {
                    self.tokenize_field(fid, field_name, value, token_fn, &mut field_position)?;
                }
                Ok(pattern_match)
            };

            // parse json.
            match serde_json::to_value(value).map_err(InternalError::SerdeJson)? {
                Value::Object(object) => seek_leaf_values_in_object(
                    &object,
                    field_name,
                    Depth::OnBaseKey,
                    &mut tokenize_field,
                )?,
                Value::Array(array) => seek_leaf_values_in_array(
                    &array,
                    field_name,
                    Depth::OnBaseKey,
                    &mut tokenize_field,
                )?,
                value => {
                    tokenize_field(field_name, Depth::OnBaseKey, &value)?;
                }
            }
        }

        Ok(())
    }

    fn tokenize_field(
        &self,
        field_id: FieldId,
        field_name: &str,
        value: &Value,
        token_fn: &mut impl FnMut(
            &str,
            u16,
            u16,
            &str,
            SearchTermKind,
        ) -> std::result::Result<(), crate::Error>,
        field_position: &mut HashMap<u16, u32>,
    ) -> Result<()> {
        let position = field_position
            .entry(field_id)
            .and_modify(|counter| *counter += MAX_DISTANCE)
            .or_insert(0);
        if *position >= self.max_positions_per_attributes {
            return Ok(());
        }

        let locales = matches!(value, Value::String(_))
            .then(|| {
                self.localized_attributes_rules
                    .iter()
                    .find(|rule| rule.match_str(field_name) == PatternMatch::Match)
                    .map(|rule| rule.locales())
            })
            .flatten();

        #[cfg(feature = "japanese")]
        let index_japanese_alternates =
            locales.is_some_and(|locales| locales.contains(&Language::Jpn));

        let text;
        let tokens = match value {
            Value::Number(n) => {
                text = n.to_string();
                self.tokenizer.tokenize(text.as_str())
            }
            Value::Bool(b) => {
                text = b.to_string();
                self.tokenizer.tokenize(text.as_str())
            }
            Value::String(text) => self.tokenizer.tokenize_with_allow_list(text.as_str(), locales),
            _ => return Ok(()),
        };

        // create an iterator of token with their positions.
        let tokens = process_tokens(*position, tokens)
            .take_while(|(p, _)| *p < self.max_positions_per_attributes);

        for (index, token) in tokens {
            // keep a word only if it is not empty and fit in a LMDB key.
            let lemma = token.lemma().trim();
            if !lemma.is_empty() && lemma.len() <= MAX_WORD_LENGTH {
                *position = index;
                if let Ok(position) = (*position).try_into() {
                    token_fn(field_name, field_id, position, lemma, SearchTermKind::Original)?;

                    #[cfg(feature = "japanese")]
                    if index_japanese_alternates {
                        if let TokenMetadata::Japanese { reading: Some(reading) } = &token.metadata
                        {
                            for alternate in
                                JapaneseSegmenter::search_alternates_from_reading(reading)
                            {
                                let alternate = alternate.trim();
                                if alternate != lemma
                                    && !alternate.is_empty()
                                    && alternate.len() <= MAX_WORD_LENGTH
                                {
                                    token_fn(
                                        field_name,
                                        field_id,
                                        position,
                                        alternate,
                                        SearchTermKind::Alternate,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// take an iterator on tokens and compute their relative position depending on separator kinds
/// if it's an `Hard` separator we add an additional relative proximity of MAX_DISTANCE between words,
/// else we keep the standard proximity of 1 between words.
fn process_tokens<'a>(
    start_offset: u32,
    tokens: impl Iterator<Item = Token<'a>>,
) -> impl Iterator<Item = (u32, Token<'a>)> {
    tokens
        .skip_while(|token| token.is_separator())
        .scan((start_offset, None), |(offset, prev_kind), mut token| {
            match token.kind {
                TokenKind::Word | TokenKind::StopWord if !token.lemma().is_empty() => {
                    *offset += match *prev_kind {
                        Some(TokenKind::Separator(SeparatorKind::Hard)) => MAX_DISTANCE,
                        Some(_) => 1,
                        None => 0,
                    };
                    *prev_kind = Some(token.kind)
                }
                TokenKind::Separator(SeparatorKind::Hard) => {
                    *prev_kind = Some(token.kind);
                }
                TokenKind::Separator(SeparatorKind::Soft)
                    if *prev_kind != Some(TokenKind::Separator(SeparatorKind::Hard)) =>
                {
                    *prev_kind = Some(token.kind);
                }
                _ => token.kind = TokenKind::Unknown,
            }
            Some((*offset, token))
        })
        .filter(|(_, t)| t.is_word())
}

/// Factorize tokenizer building.
pub fn tokenizer_builder<'a>(
    stop_words: Option<&'a fst::Set<&'a [u8]>>,
    allowed_separators: Option<&'a [&str]>,
    dictionary: Option<&'a [&str]>,
) -> TokenizerBuilder<'a, &'a [u8]> {
    let mut tokenizer_builder = TokenizerBuilder::new();
    if let Some(stop_words) = stop_words {
        tokenizer_builder.stop_words(stop_words);
    }
    if let Some(dictionary) = dictionary {
        tokenizer_builder.words_dict(dictionary);
    }
    if let Some(separators) = allowed_separators {
        tokenizer_builder.separators(separators);
    }

    tokenizer_builder
}

#[cfg(test)]
mod test {
    use bumpalo::Bump;
    use bumparaw_collections::RawMap;
    use charabia::TokenizerBuilder;
    use meili_snap::snapshot;
    use rustc_hash::FxBuildHasher;
    use serde_json::json;
    use serde_json::value::RawValue;

    use super::*;
    use crate::fields_ids_map::metadata::{FieldIdMapWithMetadata, MetadataBuilder};
    use crate::update::new::document::{DocumentFromVersions, Versions};
    use crate::{FieldsIdsMap, GlobalFieldsIdsMap, UserError};

    #[test]
    fn test_japanese_reading_and_romaji_are_same_position_alternates() {
        let mut fields_ids_map = FieldsIdsMap::new();
        let name_fid = fields_ids_map.insert("name").unwrap();
        let document = json!({ "name": "国際空港" });

        let locales = [charabia::Language::Jpn];
        let localized_rules =
            [LocalizedAttributesRule::new(vec!["name".to_string()], vec![charabia::Language::Jpn])];
        let mut tb = TokenizerBuilder::default();
        tb.allow_list(&locales);
        let tokenizer = tb.build();
        let document_tokenizer = DocumentTokenizer {
            tokenizer: &tokenizer,
            localized_attributes_rules: &localized_rules,
            max_positions_per_attributes: 1000,
        };

        let fields_ids_map = FieldIdMapWithMetadata::new(
            fields_ids_map,
            MetadataBuilder::new(
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                None,
                None,
                Default::default(),
            ),
        );
        let fields_ids_map_lock = std::sync::RwLock::new(fields_ids_map);
        let mut global_fields_ids_map = GlobalFieldsIdsMap::new(&fields_ids_map_lock);

        let document = document.to_string();
        let bump = Bump::new();
        let document: &RawValue = serde_json::from_str(&document).unwrap();
        let document = RawMap::from_raw_value_and_hasher(document, FxBuildHasher, &bump).unwrap();
        let document = Versions::single(document);
        let document = DocumentFromVersions::new(&document);

        let mut terms = Vec::new();
        document_tokenizer
            .tokenize_document(
                document,
                &mut |field_name: &str| {
                    let id = global_fields_ids_map.id_or_insert(field_name).unwrap();
                    Ok((id, PatternMatch::Match))
                },
                &mut |_field_name, fid, pos, word, term_kind| {
                    terms.push((fid, pos, word.to_string(), term_kind));
                    Ok(())
                },
            )
            .unwrap();

        let surface_pos = terms
            .iter()
            .find_map(|(fid, pos, word, alt)| {
                (*fid == name_fid && word == "国際" && *alt == SearchTermKind::Original)
                    .then_some(*pos)
            })
            .expect("surface token should be indexed");

        assert!(terms.iter().any(|(fid, pos, word, alt)| {
            *fid == name_fid
                && *pos == surface_pos
                && word == "こくさい"
                && *alt == SearchTermKind::Alternate
        }));
        assert!(terms.iter().any(|(fid, pos, word, alt)| {
            *fid == name_fid
                && *pos == surface_pos
                && word == "kokusai"
                && *alt == SearchTermKind::Alternate
        }));
    }

    #[test]
    fn test_tokenize_document() {
        let mut fields_ids_map = FieldsIdsMap::new();

        let document = json!({
            "doggo": {                "name": "doggo",
            "age": 10,},
            "catto": {
                "catto": {
                    "name": "pesti",
                    "age": 23,
                }
            },
            "doggo.name": ["doggo", "catto"],
            "not-me": "UNSEARCHABLE",
            "me-nether": {"nope": "unsearchable"}
        });

        let _field_1_id = fields_ids_map.insert("doggo").unwrap();
        let _field_2_id = fields_ids_map.insert("catto").unwrap();
        let _field_3_id = fields_ids_map.insert("doggo.name").unwrap();
        let _field_4_id = fields_ids_map.insert("not-me").unwrap();
        let _field_5_id = fields_ids_map.insert("me-nether").unwrap();

        let mut tb = TokenizerBuilder::default();
        let document_tokenizer = DocumentTokenizer {
            tokenizer: &tb.build(),
            localized_attributes_rules: &[],
            max_positions_per_attributes: 1000,
        };

        let fields_ids_map = FieldIdMapWithMetadata::new(
            fields_ids_map,
            MetadataBuilder::new(
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                None,
                None,
                Default::default(),
            ),
        );

        let fields_ids_map_lock = std::sync::RwLock::new(fields_ids_map);
        let mut global_fields_ids_map = GlobalFieldsIdsMap::new(&fields_ids_map_lock);

        let mut words = std::collections::BTreeMap::new();

        let document = document.to_string();

        let bump = Bump::new();
        let document: &RawValue = serde_json::from_str(&document).unwrap();
        let document = RawMap::from_raw_value_and_hasher(document, FxBuildHasher, &bump).unwrap();

        let document = Versions::single(document);
        let document = DocumentFromVersions::new(&document);

        let mut should_tokenize = |field_name: &str| {
            let Some(field_id) = global_fields_ids_map.id_or_insert(field_name) else {
                return Err(UserError::AttributeLimitReached.into());
            };

            Ok((field_id, PatternMatch::Match))
        };

        document_tokenizer
            .tokenize_document(
                document,
                &mut should_tokenize,
                &mut |_fname, fid, pos, word, _term_kind| {
                    words.insert([fid, pos], word.to_string());
                    Ok(())
                },
            )
            .unwrap();

        snapshot!(format!("{:#?}", words), @r###"
        {
            [
                2,
                0,
            ]: "doggo",
            [
                2,
                8,
            ]: "doggo",
            [
                2,
                16,
            ]: "catto",
            [
                3,
                0,
            ]: "unsearchable",
            [
                5,
                0,
            ]: "10",
            [
                7,
                0,
            ]: "pesti",
            [
                8,
                0,
            ]: "23",
            [
                9,
                0,
            ]: "unsearchable",
        }
        "###);
    }
}
