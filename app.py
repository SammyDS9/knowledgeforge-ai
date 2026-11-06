import streamlit as st
from knowledgeforge.core import extract_document, chunk_documents, Retriever, grounded_answer

st.set_page_config(page_title='KnowledgeForge AI', page_icon='📚', layout='wide')
st.title('📚 KnowledgeForge AI — Document Intelligence & RAG')
st.caption('Portfolio demonstration • source-grounded document search • synthetic sample data')

with st.sidebar:
    st.header('Settings')
    st.info('Use synthetic/public documents only. Do not upload PHI, confidential files, or company data.')
    key = st.text_input('OpenAI API key (optional)', type='password', help='Retrieval works without an API key. API usage may incur charges.')
    model = st.text_input('OpenAI model', value='gpt-4o-mini')
    top_k = st.slider('Number of retrieved passages', 1, 8, 4)
    st.caption('API keys entered here are not saved to GitHub. Avoid sharing them on public computers.')

files = st.file_uploader('Upload PDF, DOCX, TXT or Markdown files', type=['pdf', 'docx', 'txt', 'md'], accept_multiple_files=True)
use_sample = st.checkbox('Include synthetic example knowledge base', value=True)

@st.cache_resource(show_spinner=False)
def make_retriever(entries):
    docs = []
    for filename, contents in entries:
        docs.extend(extract_document(filename, contents))
    return Retriever(chunk_documents(docs))

entries = [(f.name, f.getvalue()) for f in files or []]
if use_sample:
    from pathlib import Path
    base = Path(__file__).parent / 'sample_docs'
    entries.extend((p.name, p.read_bytes()) for p in sorted(base.glob('*.md')))

if entries:
    try:
        retriever = make_retriever(tuple(entries))
        st.success(f'Indexed {len(entries)} document(s) into {len(retriever.chunks)} searchable passages.')
        question = st.text_input('Ask a question about your documents', placeholder='What is the escalation process for high-risk claims?')
        if st.button('Find answer', type='primary'):
            if not question.strip():
                st.warning('Enter a question first.')
            else:
                matches = retriever.search(question, top_k)
                if not matches:
                    st.warning('No relevant evidence found. Try different keywords.')
                else:
                    try:
                        with st.spinner('Retrieving and preparing answer...'):
                            answer = grounded_answer(question, matches, key, model)
                        st.subheader('Answer')
                        st.write(answer)
                    except Exception as exc:
                        st.error(f'LLM answer unavailable: {exc.__class__.__name__}. Retrieved passages remain available below.')
                    st.subheader('Supporting source passages')
                    for idx, (chunk, score) in enumerate(matches, 1):
                        label = f'[{idx}] {chunk.source}' + (f' — page {chunk.page}' if chunk.page else '')
                        with st.expander(f'{label} • relevance {score:.2f}', expanded=(idx == 1)):
                            st.write(chunk.text)
    except Exception as exc:
        st.error(f'Could not index documents: {exc}')
else:
    st.info('Upload a document or enable the synthetic example knowledge base to begin.')

st.divider()
st.caption('Architecture: document ingestion → text extraction → overlapping chunks → TF-IDF retrieval → optional grounded LLM synthesis → cited source passages. This prototype uses lexical retrieval, not an embedding vector database.')
