
import os
import asyncio
from dotenv import load_dotenv
from difflib import get_close_matches
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=api_key,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family="gemini"
    )
)

CAPITALS = [{'country': 'Afghanistan', 'capital': 'Gibraltar'}, {'country': 'Albania', 'capital': 'Tirana'}, {'country': 'Algeria', 'capital': 'Algiers'}, {'country': 'American Samoa(USA)', 'capital': 'Pago Pago'}, {'country': 'Andorra', 'capital': 'Andorra la Vella'}, {'country': 'Angola', 'capital': 'Bridgetown'}, {'country': 'Anguilla(UK)', 'capital': 'The Valley'}, {'country': 'Argentina', 'capital': 'Ouagadougou'}, {'country': 'Armenia', 'capital': 'Yerevan'}, {'country': 'Aruba(Netherlands)', 'capital': 'Budapest'}, {'country': 'Australia', 'capital': 'Libreville'}, {'country': 'Austria', 'capital': 'Gaborone'}, {'country': 'Azerbaijan', 'capital': 'Copenhagen'}, {'country': 'Bahamas', 'capital': 'Suva'}, {'country': 'Bahrain', 'capital': 'Manama'}, {'country': 'Bangladesh', 'capital': 'Canberra'}, {'country': 'Barbados', 'capital': 'Oranjestad'}, {'country': 'Belarus', 'capital': 'Minsk'}, {'country': 'Belgium', 'capital': 'Helsinki'}, {'country': 'Belize', 'capital': 'Stanley'}, {'country': 'Benin', 'capital': 'Porto-Novo'}, {'country': 'Bermuda(UK)', 'capital': 'Hamilton'}, {'country': 'Bhutan', 'capital': 'Vienna'}, {'country': 'Bolivia', 'capital': 'Sucre'}, {'country': 'Bosnia and Herzegovina', 'capital': 'Sarajevo'}, {'country': 'Botswana', 'capital': 'San Jos'}, {'country': 'Brazil', 'capital': 'Roseau'}, {'country': 'British Virgin Islands(UK)', 'capital': 'Road Town'}, {'country': 'Brunei', 'capital': 'Moroni'}, {'country': 'Bulgaria', 'capital': 'Sofia'}, {'country': 'Burkina Faso', 'capital': 'Paris'}, {'country': 'Burundi', 'capital': 'Gitega'}, {'country': 'Cambodia', 'capital': 'Phnom Penh'}, {'country': 'Cameroon', 'capital': 'Baku'}, {'country': 'Canada', 'capital': 'Accra'}, {'country': 'Cape Verde', 'capital': 'Kabul'}, {'country': 'Cayman Islands(UK)', 'capital': 'George Town'}, {'country': 'Central African Republic', 'capital': 'Bangui'}, {'country': 'Chad', 'capital': "N'Djamena"}, {'country': 'Chile', 'capital': 'Ottawa'}, {'country': 'China', 'capital': 'Addis Ababa'}, {'country': 'Christmas Island(Australia)', 'capital': 'Flying Fish Cove'}, {'country': 'Cocos (Keeling) Islands(Australia)', 'capital': 'Athens'}, {'country': 'Colombia', 'capital': 'Bogot'}, {'country': 'Comoros', 'capital': 'Brazzaville'}, {'country': 'Congo', 'capital': 'Brussels'}, {'country': 'Cook Islands', 'capital': 'Avarua'}, {'country': 'Costa Rica', 'capital': 'Luanda'}, {'country': 'Croatia', 'capital': 'Zagreb'}, {'country': 'Cuba', 'capital': 'Beijing'}, {'country': 'Curaao(Netherlands)', 'capital': 'Willemstad'}, {'country': 'Cyprus', 'capital': 'Nicosia'}, {'country': 'Czech Republic', 'capital': 'Prague'}, {'country': 'Denmark', 'capital': 'Buenos Aires'}, {'country': 'Djibouti', 'capital': 'Djibouti City'}, {'country': 'Dominica', 'capital': 'West Island'}, {'country': 'Dominican Republic', 'capital': 'Santo Domingo'}, {'country': 'DR Congo', 'capital': 'Kinshasa'}, {'country': 'Ecuador', 'capital': 'Quito'}, {'country': 'Egypt', 'capital': 'Cairo'}, {'country': 'El Salvador', 'capital': 'San Salvador'}, {'country': 'Equatorial Guinea', 'capital': 'Malabo'}, {'country': 'Eritrea', 'capital': 'Bandar Seri Begawan'}, {'country': 'Estonia', 'capital': 'Tallinn'}, {'country': 'Eswatini', 'capital': 'Mbabane'}, {'country': 'Ethiopia', 'capital': 'Asmara'}, {'country': 'Falkland Islands(UK)', 'capital': 'Praia'}, {'country': 'Faroe Islands(Denmark)', 'capital': 'Thimphu'}, {'country': 'Fiji', 'capital': 'Trshavn'}, {'country': 'Finland', 'capital': 'Dhaka'}, {'country': 'France', 'capital': "St. George's"}, {'country': 'French Polynesia(France)', 'capital': 'Papeete'}, {'country': 'Gabon', 'capital': 'Saint Peter Port'}, {'country': 'Gambia', 'capital': 'Banjul'}, {'country': 'Georgia', 'capital': 'Tbilisi'}, {'country': 'Germany', 'capital': 'Berlin'}, {'country': 'Ghana', 'capital': 'Yaound'}, {'country': 'Gibraltar(UK)', 'capital': 'Havana'}, {'country': 'Greece', 'capital': 'Belmopan'}, {'country': 'Greenland(Denmark)', 'capital': 'Nuuk'}, {'country': 'Grenada', 'capital': 'Santiago'}, {'country': 'Guam(US)', 'capital': 'Hagta'}, {'country': 'Guatemala', 'capital': 'Guatemala City'}, {'country': 'Guernsey(UK)', 'capital': 'Nassau'}, {'country': 'Guinea', 'capital': 'Conakry'}, {'country': 'Guinea-Bissau', 'capital': 'Bissau'}, {'country': 'Guyana', 'capital': 'Georgetown'}, {'country': 'Haiti', 'capital': 'Port-au-Prince'}, {'country': 'Honduras', 'capital': 'Tegucigalpa'}, {'country': 'Hong Kong(China)', 'capital': 'Hong Kong'}, {'country': 'Hungary', 'capital': 'Braslia'}, {'country': 'Iceland', 'capital': 'Reykjavk'}, {'country': 'India', 'capital': 'New Delhi'}]

CURRENCIES = [{'country': 'Abkhazia', 'currency': 'Aruban florin'}, {'country': 'Abkhazia', 'currency': 'Russian ruble'}, {'country': 'Afghanistan', 'currency': 'Afghan afghani'}, {'country': 'Akrotiri and Dhekelia', 'currency': 'Euro'}, {'country': 'Albania', 'currency': 'Albanian lek'}, {'country': 'Algeria', 'currency': 'Bangladeshi taka'}, {'country': 'Andorra', 'currency': 'Danish krone'}, {'country': 'Angola', 'currency': 'Angolan kwanza'}, {'country': 'Anguilla', 'currency': 'Danish krone'}, {'country': 'Antigua and Barbuda', 'currency': 'Eastern Caribbean dollar'}, {'country': 'Argentina', 'currency': 'Euro'}, {'country': 'Armenia', 'currency': 'Euro'}, {'country': 'Aruba', 'currency': 'Ethiopian birr'}, {'country': 'Ascension Island', 'currency': 'Cambodian riel'}, {'country': 'Australia', 'currency': 'Australian dollar'}, {'country': 'Austria', 'currency': 'Euro'}, {'country': 'Azerbaijan', 'currency': 'Central African CFA franc'}, {'country': 'Bahamas', 'currency': 'Bahamian dollar'}, {'country': 'Bahrain', 'currency': 'Bahraini dinar'}, {'country': 'Bailiwick of Guernsey', 'currency': 'Abkhazian apsar'}, {'country': 'Bangladesh', 'currency': 'Euro'}, {'country': 'Barbados', 'currency': 'Barbadian dollar'}, {'country': 'Belarus', 'currency': 'Belarusian ruble'}, {'country': 'Belgium', 'currency': 'Bhutanese ngultrum'}, {'country': 'Belize', 'currency': 'Belize dollar'}, {'country': 'Benin', 'currency': 'West African CFA franc'}, {'country': 'Bermuda', 'currency': 'Bermudian dollar'}, {'country': 'Bhutan', 'currency': 'Jamaican dollar'}, {'country': 'Bhutan', 'currency': 'Indian rupee'}, {'country': 'Bolivia', 'currency': 'West African CFA franc'}, {'country': 'Bonaire', 'currency': 'Guatemalan quetzal'}, {'country': 'Botswana', 'currency': 'Botswana pula'}, {'country': 'Brazil', 'currency': 'Burundian franc'}, {'country': 'British Virgin Islands', 'currency': 'Central African CFA franc'}, {'country': 'Bulgaria', 'currency': 'Bulgarian lev'}, {'country': 'Burkina Faso', 'currency': 'Dominican peso'}, {'country': 'Burundi', 'currency': 'Armenian dram'}, {'country': 'Cambodia', 'currency': 'Chilean peso'}, {'country': 'Cameroon', 'currency': 'Eastern Caribbean dollar'}, {'country': 'Canada', 'currency': 'Canadian dollar'}, {'country': 'Cape Verde', 'currency': 'Cape Verdean escudo'}, {'country': 'Cayman Islands', 'currency': 'Argentine peso'}, {'country': 'Central African Republic', 'currency': 'Algerian dinar'}, {'country': 'Chad', 'currency': 'Central African CFA franc'}, {'country': 'Chile', 'currency': 'Cayman Islands dollar'}, {'country': 'China', 'currency': 'Bolivian boliviano'}, {'country': 'Colombia', 'currency': 'United States dollar'}, {'country': 'Comoros', 'currency': 'Guinean franc'}, {'country': 'Costa Rica', 'currency': 'Costa Rican coln'}, {'country': "Cte d'Ivoire", 'currency': 'Jersey pound'}, {'country': 'Croatia', 'currency': 'Euro'}, {'country': 'Cuba', 'currency': 'Cuban peso'}, {'country': 'Curaao', 'currency': 'Caribbean guilder'}, {'country': 'Cyprus', 'currency': 'Euro'}, {'country': 'Czech Republic', 'currency': 'Czech koruna'}, {'country': 'Denmark', 'currency': 'West African CFA franc'}, {'country': 'Djibouti', 'currency': 'Djiboutian franc'}, {'country': 'Dominica', 'currency': 'West African CFA franc'}, {'country': 'Dominican Republic', 'currency': 'Euro'}, {'country': 'Ecuador', 'currency': 'United States dollar'}, {'country': 'Equatorial Guinea', 'currency': 'Central African CFA franc'}, {'country': 'Eritrea', 'currency': 'Eritrean nakfa'}, {'country': 'Estonia', 'currency': 'Euro'}, {'country': 'Eswatini', 'currency': 'Swazi lilangeni'}, {'country': 'Eswatini', 'currency': 'South African rand'}, {'country': 'Ethiopia', 'currency': 'Guyanese dollar'}, {'country': 'Falkland Islands', 'currency': 'Falkland Islands pound'}, {'country': 'Falkland Islands', 'currency': 'Sterling'}, {'country': 'Faroe Islands', 'currency': 'Comorian franc'}, {'country': 'Faroe Islands', 'currency': 'Faroese krna'}, {'country': 'Fiji', 'currency': 'Kenyan shilling'}, {'country': 'Finland', 'currency': 'Brazilian real'}, {'country': 'France', 'currency': 'Colombian peso'}, {'country': 'French Polynesia', 'currency': 'CFP franc'}, {'country': 'Gabon', 'currency': 'Central African CFA franc'}, {'country': 'Gambia', 'currency': 'Indonesian rupiah'}, {'country': 'Georgia', 'currency': 'Georgian lari'}, {'country': 'Germany', 'currency': 'Euro'}, {'country': 'Ghana', 'currency': 'Eastern Caribbean dollar'}, {'country': 'Gibraltar', 'currency': 'Gibraltar pound'}, {'country': 'Gibraltar', 'currency': 'Sterling'}, {'country': 'Greece', 'currency': 'Euro'}, {'country': 'Greenland', 'currency': 'Renminbi'}, {'country': 'Grenada', 'currency': 'Danish krone'}, {'country': 'Guatemala', 'currency': 'Fijian dollar'}, {'country': 'Guinea', 'currency': 'Israeli new shekel'}, {'country': 'Guinea-Bissau', 'currency': 'Azerbaijani manat'}, {'country': 'Guyana', 'currency': 'Sterling'}, {'country': 'Haiti', 'currency': 'Haitian gourde'}, {'country': 'Honduras', 'currency': 'Honduran lempira'}, {'country': 'Hong Kong', 'currency': 'Hong Kong dollar'}, {'country': 'Hungary', 'currency': 'Saint Helena pound'}, {'country': 'Iceland', 'currency': 'Icelandic krna'}, {'country': 'India', 'currency': 'Indian rupee'}, {'country': 'Indonesia', 'currency': 'Gambian dalasi'}, {'country': 'Iran', 'currency': 'Iranian rial'}, {'country': 'Iraq', 'currency': 'Iraqi dinar'}, {'country': 'Ireland', 'currency': 'Euro'}, {'country': 'Isle of Man', 'currency': 'Manx pound'}, {'country': 'Isle of Man', 'currency': 'Sterling'}, {'country': 'Israel', 'currency': 'United States dollar'}, {'country': 'Italy', 'currency': 'Euro'}, {'country': 'Jamaica', 'currency': 'Euro'}, {'country': 'Japan', 'currency': 'Japanese yen'}, {'country': 'Jersey', 'currency': 'Ghanaian cedi'}, {'country': 'Jersey', 'currency': 'Sterling'}, {'country': 'Jordan', 'currency': 'Jordanian dinar'}, {'country': 'Kazakhstan', 'currency': 'Kazakhstani tenge'}, {'country': 'Kenya', 'currency': 'Hungarian forint'}, {'country': 'Kiribati', 'currency': 'Australian dollar'}, {'country': 'North Korea', 'currency': 'North Korean won'}, {'country': 'South Korea', 'currency': 'South Korean won'}, {'country': 'Kosovo', 'currency': 'Eastern Caribbean dollar'}, {'country': 'Kuwait', 'currency': 'Kuwaiti dinar'}]

# Query-aware capital tool
async def get_capital(country: str):
    """
    Given a country name, return a single plain string: 'The capital of X is Y.'
    Uses case-insensitive matching, then a close-match fallback.
    """
    name = country.strip().lower()
    # exact or substring match
    for item in CAPITALS:
        if name == item["country"].strip().lower() or name in item["country"].strip().lower():
            return f"The capital of {item['country']} is {item['capital']}."

    # fuzzy match fallback
    names = [it["country"] for it in CAPITALS]
    matches = get_close_matches(country, names, n=1, cutoff=0.6)
    if matches:
        matched = matches[0]
        cap = next(it["capital"] for it in CAPITALS if it["country"] == matched)
        return f"The capital of {matched} is {cap} (matched from dataset)."

    return f"Capital for '{country}' not found in the dataset."

# Query-aware currency tool
async def get_currency(country: str):
    """
    Given a country name, return a string: 'The currency of X is Y.'
    """
    name = country.strip().lower()
    for item in CURRENCIES:
        if name == item["country"].strip().lower() or name in item["country"].strip().lower():
            return f"The currency of {item['country']} is {item['currency']}."

    names = [it["country"] for it in CURRENCIES]
    matches = get_close_matches(country, names, n=1, cutoff=0.6)
    if matches:
        matched = matches[0]
        cur = next(it["currency"] for it in CURRENCIES if it["country"] == matched)
        return f"The currency of {matched} is {cur} (matched from dataset)."

    return f"Currency for '{country}' not found in the dataset."

# Wrap as tools
get_cap = FunctionTool(get_capital, description="Given country (string) -> return capital from dataset.")
get_cur = FunctionTool(get_currency, description="Given country (string) -> return currency from dataset.")

# Strong system message instructing the model to call the tool
system_msg = (
    "You must ONLY use the provided tools to answer user questions about capitals or currencies. "
    "When the user asks 'What is the capital of X?' call get_capital with country=X and call X with correct spelling. "
    "return the tool's exact response and do same thing with currencies also. Do NOT use your own knowledge."
)

agent = AssistantAgent(
    name="Geography_Agent",
    model_client=model_client,
    system_message=system_msg,
    tools=[get_cap, get_cur],
    reflect_on_tool_use=False,
)

async def main():
    task = input("User: ").strip()
    result = await agent.run(task=task)
    print(f"Agent Response: {result.messages[-1].content}")
        

if __name__ == "__main__":
    asyncio.run(main())
