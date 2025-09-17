import os
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from langdetect import detect
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader


load_dotenv()
app = FastAPI()

ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  


client = Client(ACCOUNT_SID, AUTH_TOKEN)

vector_store = None
qa_chain = None

def initialize_rag_system():
    global vector_store, qa_chain
    
    try:
        print("Initializing RAG system...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  
            temperature=0.7,
            max_tokens=1024
        )

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        knowledge_file = "knowledge_base.txt"
        
        if os.path.exists(knowledge_file):
            print(f"Loading knowledge base from {knowledge_file}...")
            
            loader = TextLoader(knowledge_file, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            print(f"Created {len(texts)} text chunks")
            
            vector_store = FAISS.from_documents(texts, embeddings)
            print("Vector store created successfully")
            
            prompt_template = """You are a helpful assistant for college students answering querys related to their college. Use the following context to answer the question. 
            If you don't know the answer based on the context, say so politely.
            
            Context: {context}
            
            Question: {question}
            
            Answer in a clear, concise, and friendly manner:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 3}  
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("RAG system initialized successfully")
            return True
            
        
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return False

def smart_translate(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
    try:
        print(f"Translating using Google: {source_lang} -> {target_lang}")
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        result = translator.translate(text)
        
        if result:
            print(f"Google translation successful")
            return result
            
    except Exception as e:
        print(f"Google translator failed: {e}")
    
    print("All translators failed, returning original text")
    return text

def detect_language_smart(text: str) -> str:
    detected = detect(text)
    print(f"Initial detection: {detected}")
    
    if detected == 'hi' and text.isascii():
        print("Detected Hinglish (Hindi words in Latin script)")
        return 'hinglish'
    elif detected == 'en':
        hinglish_words = ['hai', 'hain', 'kya', 'aap', 'main', 'kar', 'kaise', 'kahan', 'kab', 'kyun']
        text_lower = text.lower()
        hinglish_count = sum(1 for word in hinglish_words if word in text_lower)
        
        if hinglish_count >= 1:  
            print("Detected Hinglish (misclassified as English)")
            return 'hinglish'
    
    return detected

def handle_hinglish(text: str, to_english: bool = True) -> str:
    try:
        if to_english:
            print("Converting Hinglish to Devanagari for translation...")
            devanagari_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            print(f"Devanagari: {devanagari_text}")
            
            english_result = smart_translate(devanagari_text, 'hi', 'en')
            return english_result
        else:
            print("Converting English to Hinglish...")
            hindi_result = smart_translate(text, 'en', 'hi')
            hinglish_result = transliterate(hindi_result, sanscript.DEVANAGARI, sanscript.ITRANS)
            return hinglish_result
            
    except Exception as e:
        print(f"Hinglish processing failed: {e}")
        return text

def process_user_query(english_text: str) -> str:
    global qa_chain
    
    try:
        if qa_chain is None:
            print("RAG system not initialized, falling back to basic response")
            return f"You asked: {english_text}\n\nI understand your message. How can I assist you further?"
        
        print(f"Processing query with Gemini RAG: {english_text}")
        
        response = qa_chain.invoke({"query": english_text})
        
        answer = response.get('result', 'I apologize, but I couldn\'t find relevant information to answer your question.')
        
        if 'source_documents' in response:
            print(f"Used {len(response['source_documents'])} source documents")
        
        print(f"Generated response: {answer[:100]}...")
        return answer
        
    except Exception as e:
        print(f"Error in RAG processing: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try again later."
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting WhatsApp Bot with Gemini and RAG...")
    success = initialize_rag_system()
    if success:
        print("WhatsApp Bot is ready!")
    else:
        print("Bot started but RAG system initialization failed")

    yield 
    print("Shutting down WhatsApp Bot...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    print(f"Message from {From}: {Body}")
    
    try:
        detected_lang = detect_language_smart(Body)
        print(f"Detected language: {detected_lang}")
        
        if detected_lang in ["sw", "id", "ms", "so", "af", "tl"]:
            detected_lang = "hinglish"
        
        if detected_lang == 'hinglish':
            print("Processing Hinglish input...")
            english_text = handle_hinglish(Body, to_english=True)
            print(f"English version: {english_text}")
            
            answer_en = process_user_query(english_text)
            
            final_answer = handle_hinglish(answer_en, to_english=False)
            
        elif detected_lang == 'en':
            print("Processing English input...")
            english_text = Body
            final_answer = process_user_query(english_text)
            
        else:
            print(f"Processing {detected_lang} input...")
            english_text = smart_translate(Body, detected_lang, 'en')
            print(f"English version: {english_text}")
            
            answer_en = process_user_query(english_text)
            
            final_answer = smart_translate(answer_en, 'en', detected_lang)
        
        return PlainTextResponse(final_answer)
        
    except Exception as e:
        print(f"Error processing message: {e}")
        resp = MessagingResponse()
        resp.message("Sorry, I encountered an error processing your message. Please try again.")
        return PlainTextResponse(str(resp))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)