import pytest
from knowledgeforge.core import Chunk, Retriever, chunk_documents, extract_document, grounded_answer

def test_text_extraction():
    docs = extract_document('guide.md', b'Claims should be reviewed weekly')
    assert docs[0].source == 'guide.md'
    assert 'Claims' in docs[0].text

def test_chunking_preserves_source():
    chunks = chunk_documents([Chunk('test.txt', 3, ' '.join(['word'] * 24))], max_words=10, overlap=2)
    assert len(chunks) == 3
    assert all(c.source == 'test.txt' and c.page == 3 for c in chunks)

def test_retrieval():
    retriever = Retriever([Chunk('a.txt', None, 'Claims denial escalation review'), Chunk('b.txt', None, 'Forecast error monitoring')])
    assert retriever.search('claims denial')[0][0].source == 'a.txt'
    assert retriever.search('zzzznotfound') == []

def test_no_key_returns_retrieval_only():
    result = grounded_answer('what?', [(Chunk('a.txt', None, 'hello'), 0.5)], '')
    assert 'Retrieval-only' in result

def test_unsupported():
    with pytest.raises(ValueError):
        extract_document('a.exe', b'bad')
