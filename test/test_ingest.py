from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lore_master.core.config import get_settings

SAMPLE = """# The Knight

The Knight is the silent protagonist of Hollow Knight, a small vessel
born of Void in the Abyss beneath the kingdom of Hallownest. It returns
to the ruined kingdom drawn by an unknown calling.

# Hallownest

Hallownest is a once-great underground kingdom ruled by the Pale King.
After the spread of the Infection, much of it fell to ruin, leaving
behind echoes of its former glory in places like the City of Tears.

# The Pale King

The Pale King was a higher being and ruler of Hallownest. He created the
vessels in an attempt to seal away the Radiance and the Infection she
spread across his kingdom.
"""

def test_splitter_produces_multiple_chunks_within_size():
    settings = get_settings()
    docs = [Document(page_content=SAMPLE, metadata={"source": "sample.md"})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= settings.chunk_size
        # Source metadata is carried through the split.
        assert chunk.metadata.get("source") == "sample.md"
