from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time


def get_blog_content(blog_url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)

    try:
        print(f"🔍 크롤링 중: {blog_url}")
        driver.get(blog_url)
        time.sleep(2)  # 페이지 로딩 대기

        driver.switch_to.frame("mainFrame")

        try:
            content_element = driver.find_element(
                By.CLASS_NAME, "se-main-container")
        except:
            content_element = driver.find_element(By.ID, "postViewArea")

        paragraphs = content_element.find_elements(By.TAG_NAME, "p")
        blog_text = "\n".join(
            [p.text for p in paragraphs if p.text.strip() != ""])

        return blog_text

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

    finally:
        driver.quit()
