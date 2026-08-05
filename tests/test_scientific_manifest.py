from app.scientific_ingest import MANIFEST_PATH, parse_manifest


def test_public_scientific_manifest_is_versioned_and_has_unique_sources():
    items = parse_manifest()
    assert "knowledge_sources" in str(MANIFEST_PATH)
    assert len(items) >= 47
    assert len({item["source_id"] for item in items}) == len(items)
    assert all(item["access"].startswith("public_") and item["url"].startswith("https://") for item in items)
    kinds = {item.get("document_kind", "legacy_paper") for item in items}
    assert {"conference_paper", "presentation", "poster", "preprint", "book_chapter"}.issubset(kinds)
