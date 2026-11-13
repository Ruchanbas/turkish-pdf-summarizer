"""Servis genelindeki limitler.

Ayrı bir modülde durur: API Lambda'sı bu değerleri okur ama pdfplumber gibi
ağır bağımlılıkları paketine almak zorunda kalmaz.
"""

#: Okunacak azami sayfa. Daha uzun belgeler kesilir ve kullanıcı uyarılır.
MAX_PAGES = 60

#: Kabul edilen azami dosya boyutu.
MAX_FILE_BYTES = 20 * 1024 * 1024
