import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger


import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama


OLLAMA_API_URL = "http://localhost:11434/api/generate"
BASE_URLs = ["https://itmo.ru/ru/", "https://abit.itmo.ru/"]
VISITED_URLS = set()
MAX_DEPTH = 3  
RAG_INDEX_PATH = "rag_index"

def fetch_page(url):
	try:
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except requests.RequestException as e:
		print(f"Ошибка при получении страницы {url}: {e}")
		return None

def parse_links(html, base_url):
	
	soup = BeautifulSoup(html, 'html.parser')
	links = set()
	for a_tag in soup.find_all('a', href=True):
		link = a_tag['href']
		if link.startswith("http"):
			links.add(link)
		elif link.startswith("/"):
			links.add(base_url + link)
	return links

def crawl(url, depth=0):
	if depth > MAX_DEPTH or url in VISITED_URLS:
		return []
	
	print(f"Парсинг: {url} (глубина: {depth})")
	VISITED_URLS.add(url)
	html = fetch_page(url)
	if not html:
		return []
	
	soup = BeautifulSoup(html, 'html.parser')
	text = soup.get_text(separator='\n')
	links = parse_links(html, BASE_URL)
	
	child_texts = []
	for link in links:
		child_texts.extend(crawl(link, depth + 1))
	
	return [text] + child_texts

def create_rag(texts):
	if os.path.exists(RAG_INDEX_PATH):
		print("Загрузка существующего индекса RAG...")
		vectorstore = FAISS.load_local(RAG_INDEX_PATH, HuggingFaceEmbeddings())
	else:
		print("Создание нового индекса RAG...")
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
		documents = text_splitter.create_documents(texts)
		vectorstore = FAISS.from_documents(documents, HuggingFaceEmbeddings())
		vectorstore.save_local(RAG_INDEX_PATH)
	
	return vectorstore

def query_ollama(prompt):
	payload = {
		"model": "llama3.2",
		"prompt": prompt,
		"stream": False
	}
	try:
		response = requests.post(OLLAMA_API_URL, json=payload, timeout=5)
		if response.status_code == 200:
			return response.json().get("response", "")
		else:
			print(f"Ошибка при запросе к Ollama: {response.status_code}")
			return ""
	except requests.Timeout:
		return "None"

texts = []
for BASE_URL in BASE_URLs:
	texts += crawl(BASE_URL)


vectorstore = create_rag(texts)
retriever = vectorstore.as_retriever()

llm = Ollama(model="llama2")
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Initialize
app = FastAPI()
logger = None

@app.on_event("startup")
async def startup_event():
	global logger
	logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
	start_time = time.time()

	body = await request.body()
	await logger.info(
		f"Incoming request: {request.method} {request.url}\n"
		f"Request body: {body.decode()}"
	)

	response = await call_next(request)
	process_time = time.time() - start_time

	response_body = b""
	async for chunk in response.body_iterator:
		response_body += chunk

	await logger.info(
		f"Request completed: {request.method} {request.url}\n"
		f"Status: {response.status_code}\n"
		f"Response body: {response_body.decode()}\n"
		f"Duration: {process_time:.3f}s"
	)

	return Response(
		content=response_body,
		status_code=response.status_code,
		headers=dict(response.headers),
		media_type=response.media_type,
	)


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
	try:
		await logger.info(f"Processing prediction request with id: {body.id}")
		# Здесь будет вызов вашей модели
		# ~ answer = 1  # Замените на реальный вызов модели
		
		prompt_reason = "Кратко ответь на вопрос, основываясь на знаниях."
		prompt_answer = "Ответь на вопрос, основываясь на знаниях. Напиши только номер ответа"
		
		
		reasoning = qa_chain.run(prompt_reason + question)
		answer = qa_chain.run(prompt_answer + question)
		
		sources: List[HttpUrl] = [
			HttpUrl(i) for i in BASE_URLs
		]

		response = PredictionResponse(
			id=body.id,
			answer=answer,
			reasoning=reasoning,
			sources=sources,
		)
		await logger.info(f"Successfully processed request {body.id}")
		return response
	except ValueError as e:
		error_msg = str(e)
		await logger.error(f"Validation error for request {body.id}: {error_msg}")
		raise HTTPException(status_code=400, detail=error_msg)
	except Exception as e:
		await logger.error(f"Internal error processing request {body.id}: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")
