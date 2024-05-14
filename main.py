import json
import os

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from pydantic import BaseModel, Field

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "customer-service"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
client = Client()

llm_model = Ollama(model="qwen")

model_name = 'maidalun1020/bce-embedding-base_v1'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True}
embed_model = HuggingFaceEmbeddings(
	model_name=model_name,
	model_kwargs=model_kwargs,
	encode_kwargs=encode_kwargs
)


class QADescription(BaseModel):
	question: str = Field(description="问题")
	answer: int = Field(description="回答")


def random_QA_pair():
	output_parser = PydanticOutputParser(pydantic_object=QADescription)
	format_instructions = output_parser.get_format_instructions()
	template_str = """你是一个{industry}行业的运营人员,请随机列举20个与{industry}行业相关的问答对

	{format_instructions}
	"""

	# 根据模板创建提示，同时在提示中加入输出解析器的说明
	prompt = PromptTemplate(
		template=template_str,
		input_variables=["industry"],
		partial_variables={"format_instructions": format_instructions})

	qa_chain = prompt | llm_model
	output = qa_chain.invoke({"industry": "供热"})
	return output.content


def save_store(index, input_qa_list):
	string_list = [json.dumps(qa, ensure_ascii=False) for qa in input_qa_list]
	print(string_list)
	db = FAISS.from_texts(string_list, embed_model)
	db.save_local(index)


def qa_query(index, question):
	db = FAISS.load_local(index, embed_model, allow_dangerous_deserialization=True)
	retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2, "k": 3})
	related_passages = retriever.invoke(question)
	print(related_passages)


# qa_list = random_QA_pair()
# qa_list = [
# 	{"question": "锅炉的主要功能是什么？", "answer": "产生热水或蒸汽用于加热目的。"},
# 	{"question": "最常见的供暖系统类型是什么？", "answer": "强制空气供暖系统。"},
# 	{"question": "锅炉和熔炉有什么区别？", "answer": "锅炉产生热水，而熔炉产生热空气。"},
# 	{"question": "恒温器的用途是什么？", "answer": "调节建筑物内的温度。"},
# ]
# save_store("customer_index", qa_list)
qa_query("customer_index", "锅炉的功能有哪些")
