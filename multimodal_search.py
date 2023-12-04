import base64
import io
from pathlib import Path
from PIL import Image

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from japanese_clip import JapaneseCLIPEmbeddings

def img_from_base64(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

data_dir = Path("data")
chroma_dir = Path("chroma_db")
output_dir = Path("temp")
output_dir.mkdir(exist_ok=True)

embeddings = JapaneseCLIPEmbeddings(device_map="cuda", cache_folder="./models") 
vectorstore = Chroma(
    collection_name="retrieval_JapaneseCLIP", 
    embedding_function=embeddings,
    persist_directory=str(chroma_dir)
)
if not chroma_dir.exists():
    loader = DirectoryLoader(str(data_dir), glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    # CLIP text encoder can only handle 77 tokens at a time
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embeddings.tokenizer,
        chunk_size=77,
        chunk_overlap=10,
    )
    split_docs = splitter.split_documents(docs)
    image_uris = list(map(
        lambda x: str(x), set(*data_dir.glob("*.jpg"), *data_dir.glob("*.png"))
    ))
    vectorstore.add_images(image_uris)
    vectorstore.add_documents(split_docs)

test_query = ""
docs_and_scores = vectorstore.similarity_search_with_score(test_query, k=10)
for i, (doc, score) in enumerate(docs_and_scores):
    print(f"Rank {i+1:02d}: (score: {score:.4f})")
    doc = doc.page_content
    # since we use base64 encoding for image, we need to decode it
    if is_base64(doc):
        img = img_from_base64(doc)
        img.save(output_dir / f"{i:03d}.jpg")
    else:
        with open(output_dir / f"{i:03d}.txt", "w") as f:
            f.write(doc)
