# RAG Based Research Assistant
This project leverages Retrieval-Augmented Generation (RAG) to streamline the extraction of contextual insights from research papers. By integrating the `OpenAI API` with `Langchain` and `Chroma DB`, the system retrieves relevant content from a curated document base and enhances user queries with accurate, context-rich responses. Designed to support researchers, academics, and professionals, the application simplifies literature review and knowledge discovery by providing concise, targeted insights from complex scientific texts.

## Setup
- First, create a python virtual environment in the project root directory with the following command (You can provide any name for the env. I have given the default name 'venv'): <br />
```
python3 -m venv venv
```
- I'm using a Linux terminal. If you are using Windows, you may use the keyword `python` instead of `python3`
- Activate the virtual environment
```
source venv/bin/activate
```
- For Windows, activation script will reside in path `venv/Scripts/activate`
- Now install the required pip packages. I have included them in `requirements.txt` file for convenience.
```
pip install -r requirements.txt
```
*Note*: Some packages I've used may have been deprecated, but still usable.
- Create a file named `.env` in the project root directory.
- It's time for you to obtain an API Key for the OpenAI API. You can create one using the following link: https://platform.openai.com/account/api-keys
- Once the key is generated, make sure you copy it and save it in your `.env` file as follows:
```
OPENAI_API_KEY="<API_KEY>"
```

## Execute
- Create a folder named `pdfs` in the project root.
- Add your research papers here.
- You are all set now. Execute the program.
```
python3 loader.py
```
