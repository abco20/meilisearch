//! Language-level Japanese lexical variants shared by indexing and query parsing.

use charabia::segmenter::JapaneseSegmenter;
use charabia::{Token, TokenMetadata};

fn contains_han(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(
            c as u32,
            0x3400..=0x4DBF
                | 0x4E00..=0x9FFF
                | 0xF900..=0xFAFF
                | 0x3005
                | 0x3007
        )
    })
}

fn contains_kana(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(
            c as u32,
            0x3040..=0x309F
                | 0x30A0..=0x30FF
                | 0x31F0..=0x31FF
                | 0x1B000..=0x1B16F
        )
    })
}

fn prolonged_vowel(previous: char) -> Option<char> {
    const A: &str = "あかがさざただなはばぱまゃやらゎわぁ";
    const I: &str = "いきぎしじちぢにひびぴみりゐぃ";
    const U: &str = "うくぐすずつづぬふぶぷむゅゆるゔぅ";
    const E: &str = "えけげせぜてでねへべぺめれゑぇ";
    const O: &str = "おこごそぞとどのほぼぽもょよろをぉ";

    if A.contains(previous) {
        Some('あ')
    } else if I.contains(previous) {
        Some('い')
    } else if U.contains(previous) {
        Some('う')
    } else if E.contains(previous) {
        Some('え')
    } else if O.contains(previous) {
        // Charabia's Japanese normalization commonly expands the explicit
        // katakana long mark on the o-row as う (e.g. コール -> こうる).
        Some('う')
    } else {
        None
    }
}

pub(crate) fn normalize_compact_reading(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut previous_kana = None;

    for c in text.chars() {
        if c == 'ー' {
            if let Some(vowel) = previous_kana.and_then(prolonged_vowel) {
                normalized.push(vowel);
                previous_kana = Some(vowel);
            } else {
                normalized.push(c);
            }
            continue;
        }

        normalized.push(c);
        // Keep the previous base kana across combining dakuten/handakuten.
        if contains_kana(&c.to_string()) && !matches!(c, '\u{3099}' | '\u{309A}') {
            previous_kana = Some(c);
        }
    }

    normalized
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SearchVariants<T> {
    pub normalized: Option<T>,
    pub phonetic: Option<T>,
    pub romaji: Option<T>,
}

impl<T> Default for SearchVariants<T> {
    fn default() -> Self {
        Self { normalized: None, phonetic: None, romaji: None }
    }
}

impl<T> SearchVariants<T> {
    pub(crate) fn map<U>(self, mut f: impl FnMut(T) -> U) -> SearchVariants<U> {
        SearchVariants {
            normalized: self.normalized.map(&mut f),
            phonetic: self.phonetic.map(&mut f),
            romaji: self.romaji.map(f),
        }
    }
}

impl<T: Eq> SearchVariants<T> {
    pub(crate) fn iter_unique(&self) -> impl Iterator<Item = &T> {
        let mut values = Vec::with_capacity(3);
        for value in [&self.normalized, &self.phonetic, &self.romaji].into_iter().flatten() {
            if !values.contains(&value) {
                values.push(value);
            }
        }
        values.into_iter()
    }
}

pub(crate) fn search_variants(token: &Token<'_>) -> SearchVariants<String> {
    let lemma = token.lemma().trim();
    let surface_normalized =
        (!contains_han(lemma) && contains_kana(lemma)).then(|| lemma.to_owned());
    let TokenMetadata::Japanese { reading: Some(reading) } = &token.metadata else {
        return SearchVariants { normalized: surface_normalized, ..Default::default() };
    };

    let mut phonetic = None;
    let mut romaji = None;
    for alternate in JapaneseSegmenter::search_alternates_from_reading(reading) {
        let alternate = alternate.trim();
        if alternate.is_empty() {
            continue;
        }
        if alternate.is_ascii() {
            romaji.get_or_insert_with(|| alternate.to_owned());
        } else {
            phonetic.get_or_insert_with(|| alternate.to_owned());
        }
    }

    let normalized = if contains_han(lemma) { phonetic.clone() } else { surface_normalized };
    SearchVariants { normalized, phonetic, romaji }
}

#[cfg(test)]
mod tests {
    use super::normalize_compact_reading;

    #[test]
    fn compact_reading_expands_prolonged_sound_marks() {
        assert_eq!(normalize_compact_reading("くーる"), "くうる");
        assert_eq!(normalize_compact_reading("すーぱー"), "すうぱあ");
        assert_eq!(normalize_compact_reading("ぱーてぃー"), "ぱあてぃい");
    }
}
