from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

#from bs4 import BeautifulSoup
import urllib.parse
import time
import pickle
import logging

options = Options()
options.add_argument('-headless')

logger = logging.getLogger('quora_answer')
logger.setLevel(logging.DEBUG)
ch_logger = logging.StreamHandler()
ch_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch_logger.setFormatter(formatter)
logger.addHandler(ch_logger)

delay = 3

def search_qn(driver, query):
  link = f'https://www.quora.com/search?q={urllib.parse.quote_plus(query)}&type=question'
  driver.get(link)
  elements = driver.find_elements_by_xpath("//div/a[@class='question_link']")
  return [element.get_attribute("href") for element in elements]


def get_answer(query):
  driver = webdriver.Chrome(options=options)
  driver.implicitly_wait(10)
  attempts = 1
  links = None
  while attempts < 10:
    logger.info(f'Attempt {attempts}/10 of getting links to answers')
    try:
      links = search_qn(driver, query)
      #print("Retrieved links")
      logger.info("Retrieved links to answers")
      break
    except Exception as e:
      logger.info(f'Failed attempt {attempts}')
      logger.error(f'{e}')
      #print("Error:", e)
      #tb = traceback.format_exc()
      #print(tb)
      attempts += 1
      time.sleep(3)

  if links is None:
    # ran out all attempts
    logger.error("Failed to retrieve links")
    #print("Failed to retrieve links from quora.com")
    driver.quit()
    return "FAIL", ""

  for link in links:
    logger.info(f'Checking {link}')
    # make sure is still on quora
    if link[:22] == "https://www.quora.com/":
      driver.get(link)
    else:
      logger.warning(f"Link {link} is not on Quora.com!")
      continue
    
    try:
      answer_count = driver.find_element_by_xpath("//div/div[contains(@class, 'answer_count')]")
      if answer_count.text == "0 Answers":
        logger.info(f"No answers found on {link}")
        continue
      logger.info(f'{answer_count.text} Found')
    except Exception as e:
      # something went wrong
      logger.error("Cannot retrieve answer count")
      logger.error(f'{e}')
      #tb = traceback.format_exc()
      #print(tb)
      continue

    for _ in range(3):
      try:
        # try to click to view more
        driver.find_element_by_xpath("//a[contains(@class, 'ui_qtext_more_link')]").click();
        driver.find_element_by_xpath("//span[.='Continue Reading']").click();
        logger.info("Cliked to view whole answer")
        break
      except Exception as e:
        try:
          driver.find_element_by_xpath("//span[.='Continue Reading']").click();
          logger.info("Cliked to view whole answer")
          break
        except Exception as e:
          # maybe dont have to click
          logger.warning("Could not click to view more")
          #logger.info("No need to click to view more")
          #time.sleep(1)

    #soup = BeautifulSoup(driver.page_source, 'html.parser')
    try:
      # try to return answer text
      answer_text = driver.find_element_by_xpath("//div[contains(@class, 'ui_qtext_expanded')]").text
      driver.quit()
      return answer_text, link
    except Exception as e:
      # something went wrong
      driver.save_screenshot("screenshot.png")
      logger.error("Could not get answer")
      logger.error(f'{e}')
      #tb = traceback.format_exc()
      #print(tb)
      continue

  # whyyy :'(
  logger.error("Failed to retrieve any answer")
  driver.quit()
  return "FAIL :'(", ""


if __name__=="__main__":
  query = input(">>> ")
  print(get_answer(query))
