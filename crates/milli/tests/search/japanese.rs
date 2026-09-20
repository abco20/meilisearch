use bumpalo::Bump;
use charabia::Language;
use heed::EnvOpenOptions;
use http_client::policy::IpPolicy;
use milli::documents::mmap_from_objects;
use milli::progress::Progress;
use milli::update::new::indexer;
use milli::update::{IndexerConfig, MissingDocumentPolicy, Settings};
use milli::vector::RuntimeEmbedders;
use milli::{
    CreateOrOpen, Index, LocalizedAttributesRule, MustStopProcessing, Object, Search,
    TermsMatchingStrategy,
};
use serde_json::{from_value, json};
use tempfile::tempdir;

fn setup_japanese_index(exact_title: bool) -> Index {
    let tmp = tempdir().unwrap();
    let options = EnvOpenOptions::new();
    let mut options = options.read_txn_without_tls();
    options.map_size(10 * 1024 * 1024);
    let index = Index::new(options, tmp.path(), CreateOrOpen::create_without_shards()).unwrap();

    let config = IndexerConfig::default();
    let mut wtxn = index.write_txn().unwrap();
    let mut settings = Settings::new(&mut wtxn, &index, &config);
    settings.set_searchable_fields(vec!["title".to_string()]);
    settings.set_localized_attributes_rules(vec![LocalizedAttributesRule::new(
        vec!["title".to_string()],
        vec![Language::Jpn],
    )]);
    if exact_title {
        settings.set_exact_attributes(["title".to_string()].into_iter().collect());
    }
    settings
        .execute(
            &MustStopProcessing::default(),
            &Progress::default(),
            &IpPolicy::danger_always_allow(),
            Default::default(),
        )
        .unwrap();
    wtxn.commit().unwrap();

    let documents: Vec<Object> = [
        json!({ "id": 0, "title": "橋" }),
        json!({ "id": 1, "title": "箸" }),
        json!({ "id": 2, "title": "国際空港" }),
        json!({ "id": 3, "title": "はし" }),
        json!({ "id": 4, "title": "橋 国際空港" }),
        json!({ "id": 5, "title": "橋 国内空港" }),
        json!({ "id": 6, "title": "新 国際空港 駅" }),
        json!({ "id": 7, "title": "新 道路 国際空港 駅" }),
        json!({ "id": 8, "title": "箸 国際空港" }),
        json!({ "id": 9, "title": "きのこいぬ" }),
    ]
    .into_iter()
    .map(|document| from_value(document).unwrap())
    .collect();
    let documents = mmap_from_objects(documents);

    let rtxn = index.read_txn().unwrap();
    let mut wtxn = index.write_txn().unwrap();
    let db_fields_ids_map = index.fields_ids_map(&rtxn).unwrap();
    let mut new_fields_ids_map = db_fields_ids_map.clone();
    let mut operations = indexer::IndexOperations::new();
    operations.replace_documents(&documents, MissingDocumentPolicy::default()).unwrap();

    let indexer_alloc = Bump::new();
    let (document_changes, operation_stats, primary_key) = operations
        .into_changes(
            &indexer_alloc,
            &index,
            &rtxn,
            None,
            &mut new_fields_ids_map,
            &MustStopProcessing::default(),
            Progress::default(),
            None,
        )
        .unwrap();
    assert!(operation_stats.into_iter().all(|stat| stat.error.is_none()));

    indexer::index(
        &mut wtxn,
        &index,
        &milli::ThreadPoolNoAbortBuilder::new().build().unwrap(),
        config.grenad_parameters(),
        &db_fields_ids_map,
        new_fields_ids_map,
        primary_key,
        &document_changes,
        RuntimeEmbedders::default(),
        &MustStopProcessing::default(),
        &Progress::default(),
        &IpPolicy::danger_always_allow(),
        &Default::default(),
    )
    .unwrap();
    wtxn.commit().unwrap();
    drop(rtxn);

    index
}

fn search_result(index: &Index, query: &str) -> milli::SearchResult {
    let txn = index.read_txn().unwrap();
    let fields_ids_map = index.fields_ids_map(&txn).unwrap();
    let progress = Progress::default();
    let mut search = Search::new(
        &txn,
        index,
        &fields_ids_map,
        "test",
        time::OffsetDateTime::now_utc(),
        &progress,
    );
    search.query(query);
    search.limit(10);
    search.terms_matching_strategy(TermsMatchingStrategy::default());
    search.execute().unwrap()
}

fn search(index: &Index, query: &str) -> Vec<u32> {
    search_result(index, query).documents_ids
}

#[test]
fn japanese_surface_form_beats_homophone_variant() {
    let index = setup_japanese_index(false);
    let result = search(&index, "橋");
    assert_eq!(result.first(), Some(&0));
    assert!(!result.contains(&1));
}

#[test]
fn japanese_reading_and_romaji_match_indexed_variants() {
    let index = setup_japanese_index(false);
    let reading = search(&index, "はし");
    assert!(reading.contains(&0));
    assert!(reading.contains(&1));
    let romaji = search(&index, "hashi");
    assert!(romaji.contains(&0));
    assert!(romaji.contains(&1));
}

#[test]
fn japanese_query_resegmentation_matches_surface_form() {
    let index = setup_japanese_index(false);
    assert!(search(&index, "こくさいくうこう").contains(&2));
    assert!(search(&index, "kokusai kuukou").contains(&2));
    assert!(search(&index, "kokusaikuukou").contains(&2));
}

#[test]
fn japanese_resegmentation_does_not_persist_compact_span_terms() {
    let index = setup_japanese_index(false);
    let txn = index.read_txn().unwrap();
    assert!(index.contains_word(&txn, "こくさい").unwrap());
    assert!(index.contains_word(&txn, "くうこう").unwrap());
    assert!(!index.contains_word(&txn, "こくさいくうこう").unwrap());
    assert!(!index.contains_word(&txn, "kokusaikuukou").unwrap());
    drop(txn);
    assert!(search(&index, "こくさいくうこう").contains(&2));
    assert!(search(&index, "kokusaikuukou").contains(&2));
}

#[test]
fn japanese_resegmentation_tries_multiple_indexed_segmentations() {
    let index = setup_japanese_index(false);

    // The globally-shortest indexed split can be `きの | こいぬ`, even though
    // that phrase does not occur in the target document. A later bounded DP
    // candidate, `きのこ | いぬ`, must still be tried.
    assert!(search(&index, "キノコイヌ").contains(&9));
}

#[test]
fn japanese_fallback_variant_can_match_originals_in_exact_attributes() {
    let index = setup_japanese_index(true);
    let result = search(&index, "ハシ");
    assert!(result.contains(&0));
    assert!(result.contains(&1));
    assert!(result.contains(&3));
}

#[test]
fn japanese_quoted_phrase_uses_query_resegmentation() {
    let index = setup_japanese_index(false);
    assert!(search(&index, "\"こくさいくうこう\"").contains(&2));
}

#[test]
fn japanese_surface_quoted_phrase_does_not_expand_homophones() {
    let index = setup_japanese_index(false);
    let result = search(&index, "\"橋\"");
    assert!(result.contains(&0));
    assert!(!result.contains(&1));
}

#[test]
fn japanese_negative_surface_word_does_not_exclude_homophones() {
    let index = setup_japanese_index(false);
    let result = search(&index, "国際空港 -橋");
    assert!(!result.contains(&4));
    assert!(result.contains(&8));
}

#[test]
fn japanese_negative_quoted_phrase_uses_query_resegmentation() {
    let index = setup_japanese_index(false);
    let result = search(&index, "橋 -\"こくさいくうこう\"");
    assert!(result.contains(&0));
    assert!(!result.contains(&4));
}

fn proximity_rank(result: &milli::SearchResult, docid: u32) -> milli::score_details::Rank {
    let pos = result.documents_ids.iter().position(|candidate| *candidate == docid).unwrap();
    result.document_scores[pos]
        .iter()
        .find_map(|detail| match detail {
            milli::score_details::ScoreDetails::Proximity(rank) => Some(*rank),
            _ => None,
        })
        .unwrap()
}

#[test]
fn japanese_resegmentation_keeps_adjacent_proximity() {
    let index = setup_japanese_index(false);
    let variant = search_result(&index, "新 こくさいくうこう 駅");
    let surface = search_result(&index, "新 国際空港 駅");
    assert_eq!(proximity_rank(&variant, 6).local_score(), 1.0);
    assert_eq!(proximity_rank(&surface, 6).local_score(), 1.0);
    assert_eq!(variant.documents_ids.first(), Some(&6));
}
