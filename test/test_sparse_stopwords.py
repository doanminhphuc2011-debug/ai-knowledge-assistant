from rag.sparse_retriever import _load_stopwords, _tokenize, reload_stopwords


def test_stopwords_are_loaded_from_file():
    reload_stopwords()
    words = _load_stopwords()
    assert "khong" in words
    assert "xin" in words


def test_tokenize_removes_stopwords_but_keeps_content_terms():
    tokens = _tokenize("Xin cho tôi một ly bạc xỉu không?")
    assert "xin" not in tokens
    assert "cho" not in tokens
    assert "toi" not in tokens
    assert "mot" not in tokens
    assert "khong" not in tokens
    assert "bac" in tokens
    assert "xiu" in tokens
