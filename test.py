import requests
import time

def send_request():
    url_orders = "http://localhost:3002/api/v1/orders"
    url_payments = "http://localhost:3003/api/v1/payments"
    url_products = "http://localhost:3004/api/v1/products"

    try:
        response1 = requests.get(url=url_orders)
        response2 = requests.get(url=url_payments)
        response3 = requests.get(url=url_products)


    except requests.exceptions.ConnectionError:
        print(f"Ошибка: Не удалось подключиться к {url_orders}")
        print("Убедитесь, что сервер запущен на localhost:3002")
        print("-" * 50)
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == '__main__':
    try:
        while True:
            send_request()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Остановлено пользователем")


