"""Model katmanı: tembel yükleme, özetleme (map-reduce), anahtar kelime.

Ağır bağımlılıklar (torch, transformers, keybert) yalnızca model gerçekten
yüklenirken içe aktarılır; ``pip install ozetle`` bunları gerektirmez.
Kurulum: ``pip install -e ".[models]"``.
"""
