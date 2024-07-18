from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class RagWorker:
    def __init__(self):
        """Инициализация"""

        self.dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", cache_dir=r"d:\hugging_face\datasets") # Загружаем датасет
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') # Загружаем энкодер

        self.data = self.dataset['train'].map(self.embed, batched=True, batch_size=16) # Эмбеддим данные

        self.data = self.data.add_faiss_index("embeddings") # Добавляем индексацию для поиска по данным

        bot_model_name = "IlyaGusev/saiga_llama3_8b" # Загружаем LLM, которая будет генерировать ответ на текст
        self.tokenizer_bot = AutoTokenizer.from_pretrained(bot_model_name, cache_dir=r"d:\hugging_face\cache")
        self.model_bot = AutoModelForCausalLM.from_pretrained(bot_model_name, cache_dir=r"d:\hugging_face\cache",
                                                              device_map="auto")
        self.generation_config = GenerationConfig.from_pretrained(bot_model_name)

        self.system_promt = """Ты - ВикиБот, русскоязычный автоматический ассистент. Тебе будут предоставлены извлеченные части длинного документа и вопрос. 
                                Ответь на него в разговорной форме. Если ты не знаешь ответа, просто скажи "Я не знаю". Не придумывай ответ."""

    def embed(self, batch):
        """Функция формирования эмбеддингов"""

        text = batch["text"]

        return {
            "embeddings": self.encoder.encode(text)
        }

    def create_answer(self, query, context):
        """Функция создание ответа на ворос"""

        promt = f"Вопрос: {query}. Контекст: {context}"

        message = self.tokenizer_bot.apply_chat_template([{
                    "role": "system",
                    "content": self.system_promt
                }, {
                    "role": "user",
                    "content": promt
                }], tokenize=False, add_generation_prompt=True)

        data = self.tokenizer_bot(message, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model_bot.device) for k, v in data.items()}

        output_ids = self.model_bot.generate(**data, generation_config=self.generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer_bot.decode(output_ids, skip_special_tokens=True).strip()

        return output

    def search(self, query, k=3):
        """Функция поиска по нашим данным. Есть возможность настроить поиск по scores (оценка близости текстов),
        т.е. если ниже определенного порога, то модель "Не знает" ответа на вопрос"""

        embedded_query = self.encoder.encode(query) # Энкодим запрос

        scores, result = self.data.get_nearest_examples(
            "embeddings", embedded_query,
            k=k
        ) # Ищем k ближайших текстов к запросу

        answer = self.create_answer(query, result["text"][0]) # Выбираем наиболее релевантный ответ

        return answer