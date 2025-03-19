import re
import inspect
import requests
import pandas as pd
import concurrent.futures

from typing import List
from utils import inference_logger
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from fake_useragent import UserAgent
from serpapi import GoogleSearch
from bs4 import BeautifulSoup

#my functions
from tools.F_predict.F_main import F_main

@tool
def code_interpreter(code_markdown: str) -> dict | str:
    """
    Execute the provided Python code string on the terminal using exec.

    The string should contain valid, executable and pure Python code in markdown syntax.
    Code should also import any required Python packages.

    Args:
        code_markdown (str): The Python code with markdown syntax to be executed.
            For example: ```python\n<code-string>\n```

    Returns:
        dict | str: A dictionary containing variables declared and values returned by function calls,
            or an error message if an exception occurred.

    Note:
        Use this function with caution, as executing arbitrary code can pose security risks.
    """
    try:
        # Extracting code from Markdown code block
        code_lines = code_markdown.split('\n')[1:-1]
        code_without_markdown = '\n'.join(code_lines)

        # Create a new namespace for code execution
        exec_namespace = {}

        # Execute the code in the new namespace
        exec(code_without_markdown, exec_namespace)

        # Collect variables and function call results
        result_dict = {}
        for name, value in exec_namespace.items():
            if callable(value):
                try:
                    result_dict[name] = value()
                except TypeError:
                    # If the function requires arguments, attempt to call it with arguments from the namespace
                    arg_names = inspect.getfullargspec(value).args
                    args = {arg_name: exec_namespace.get(arg_name) for arg_name in arg_names}
                    result_dict[name] = value(**args)
            elif not name.startswith('_'):  # Exclude variables starting with '_'
                result_dict[name] = value

        return result_dict

    except Exception as e:
        error_message = f"An error occurred: {e}"
        inference_logger.error(error_message)
        return error_message

@tool
def google_search_and_scrape(query: str) -> dict:
    """
    Performs a Google search for the given query, retrieves the top search result URLs,
    and scrapes the text content and table data from those pages in parallel. 
    Use this function whenever the user asks for information that is not explicitly available within your knowledge base. 

    Args:
        query (str): The search query provided by the user.
        
    Returns:
        list: A list of dictionaries containing the URL, text content, and table data for each scraped page.
    """
    num_results = 2
    url = 'https://www.google.com/search'
    params = {'q': query, 'num': num_results}
    api_key = "44d8c282748defc8e8e705fd248d49493e9dc78d30682749811a7920d2b11cfc"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results
    }

    search = GoogleSearch(params)
    response = search.get_dict()
    
    # 검색 결과에서 URL 추출
    urls = [item["link"] for item in response.get("organic_results", [])[:num_results]]

    # 병렬 크롤링 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(lambda url: (url, requests.get(url, headers=headers).text if isinstance(url, str) else None), url) for url in urls[:num_results] if isinstance(url, str)]
        results = []
        for future in concurrent.futures.as_completed(futures):
            url, html = future.result()
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
            text_content = ' '.join(paragraphs)
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = text_content[:1000]
            table_data = [[cell.get_text(strip=True) for cell in row.find_all('td')] for table in soup.find_all('table') for row in table.find_all('tr')]
            if text_content or table_data:
                results.append({'url': url, 'content': text_content, 'tables': table_data})
    return results

@tool
def CAS_to_SMILES(cas : str) -> str:
    """
    Retrieve the SMILES representation of a chemical compound given its CAS number.
    Use this function whenever the user requests a SMILES from a CAS number.
    
    Args:
        cas (str): The CAS Registry Number of the chemical compound. 

    Returns:
        str: The SMILES representation of the given chemical compound.
    """

    ua = UserAgent()
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    header = {"User-Agent":ua.random}
    
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/isomericsmiles/txt'
    r = requests.get(url, verify=False, headers = header, timeout=100)
    if r.status_code != 200:
        return None
    r.encoding = 'utf-8'
    smiles = r.text.strip()
    smiles = smiles.split('\n')[0]
    return smiles     

@tool
def SMILES_to_CID(smiles : str) -> str:
    """
    Retrieve the PubChem Compound ID (CID) for a given SMILES string.
    Use this function whenever the user requests a PubChem CID from a SMILES string.
    
    Args:
        smiles (str): The SMILES representation of the chemical compound.

    Returns:
        str: The PubChem CID of the given chemical compound.
    """
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/txt'
    ua = UserAgent()
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    header = {"User-Agent":ua.random}
    r = requests.get(url, verify=False, headers = header, timeout=10)
    if r.status_code != 200:
        return None
    r.encoding = 'utf-8'
    cid = r.text.strip()
    cid = cid.split('\n')[0]
    return cid

@tool
def CID_to_NAME(cid : str) -> str:
    """
    Retrieve the chemical compound name for a given PubChem Compound ID (CID).
    Use this function whenever the user requests the chemical name from a PubChem CID.
    
    Args:
        cid (str): The PubChem Compound ID of the chemical compound.

    Returns:
        str: The common or IUPAC name of the given chemical compound.    
    """
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/iupacname/txt'
    ua = UserAgent()
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    header = {"User-Agent":ua.random}
    r = requests.get(url, verify=False, headers = header, timeout=10)
    if r.status_code != 200:
        return None
    r.encoding = 'utf-8'
    name = r.text.strip()
    return name

@tool
def SMILES_to_USE(smiles:str) -> list:
    """
    Predicts the functional uses of a given chemical compounds based on their SMILES representations.
    Use this function whenever the user requests functional uses of chemical.
    
    Args:
        smiles (str): SMILES strings.

    Returns:
        list: A list where each element corresponds to the predicted functions of the respective SMILES string in the input list, 
        each function prediction is represented as a single string containing multiple functional categories separated by a semicolon (';'). 
    """
    function_list = F_main([smiles])
    return function_list

def get_openai_tools() -> List[dict]:
    functions = [
        # code_interpreter,
        google_search_and_scrape,
        CAS_to_SMILES,
        SMILES_to_CID,
        CID_to_NAME,
        SMILES_to_USE
    ]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools
