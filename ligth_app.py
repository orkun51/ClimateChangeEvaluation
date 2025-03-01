import os
import sys
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# EXE içinden model dosyasını doğru bulmak için:
if getattr(sys, 'frozen', False):  # Eğer PyInstaller ile paketlenmişse
    model_path = os.path.join(sys._MEIPASS, "quantized_model.tflite")
else:
    model_path = "quantized_model.tflite"

# 📥 **Modeli Yükleme Fonksiyonu**
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

# 🎨 **Kategori Açıklamaları**
kategori_aciklamalari = {
    0: "Karbon Salınımı (Fabrika ve Araba): Hava kirliliği, sanayi bacalarından çıkan dumanlar veya yoğun araç trafiği olabilir.",
    1: "Ağaçlar ve Doğa: Yeşil alanlar, temiz hava ve doğayla ilgili unsurlar içeriyor olabilir.",
    2: "Güneş ve Sıcaklık Değişimi: Güneşin etkisini, sıcak hava dalgalarını veya eriyen buzulları içerebilir.",
    3: "Hava Kirliliği: Egzoz dumanları, kirli hava ve çevresel kirlilik unsurları içerebilir.",
    4: "Kuraklık ve Susuzluk: Kuruyan topraklar, su kıtlığı çeken insanlar veya su kaynaklarının azalmasını içerebilir.",
    5: "Buzulların Erimesi: Küresel ısınmanın etkisiyle eriyen buzullar ve yükselen deniz seviyesi olabilir.",
    6: "İklim Değişikliği Etkileri: Doğal afetler, aşırı hava olayları ve ekolojik tahribat unsurlarını içerebilir."
}

# 🎨 **Öğrencilere Çizim Önerileri**
cizim_onerileri = {
    0: "Sanayi bacalarından çıkan dumanları veya egzoz dumanlarını belirginleştirerek kirliliği vurgulayabilirsin.",
    1: "Daha fazla ağaç, çiçek veya temiz su kaynakları ekleyerek doğanın güzelliğini ön plana çıkarabilirsin.",
    2: "Güneş ışınlarını daha belirgin çizerek sıcaklık etkisini gösterebilirsin. Ayrıca terleyen insanlar eklemek de etkili olabilir!",
    3: "Hava kirliliğini göstermek için dumanlı bir şehir manzarası veya maske takan insanlar çizebilirsin.",
    4: "Kurumuş göller, çatlamış toprak veya susuzluktan etkilenen bitkiler ekleyerek daha güçlü bir mesaj verebilirsin.",
    5: "Eriyen buz kütleleri ve sulara düşen buz parçalarını çizerek küresel ısınmanın etkisini vurgulayabilirsin.",
    6: "Kasırgalar, fırtınalar veya orman yangınları gibi iklim değişikliğinin yol açtığı afetleri çizebilirsin."
}

# 🖌 **Çizim Değerlendirme Uygulaması Başlığı**
st.markdown("<h1 style='text-align: center;'>🎨 İklim Değişikliği Çizim Değerlendirme Uygulaması</h1>", unsafe_allow_html=True)
st.write("Bir çizim yükleyerek değerlendirme sonucunu görebilirsiniz.")

# 📂 **Dosya Yükleme**
uploaded_file = st.file_uploader("📤 Çiziminizi yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 🖼 **Görseli Göster**
    image = Image.open(uploaded_file)
    st.image(image, caption="🖌 Yüklenen Çizim", use_container_width=True)

    # 📌 **Görseli İşleme**
    img = image.convert("RGB").resize((224, 224))  # Modelin beklediği boyut
    img_array = np.array(img) / 255.0  # Normalizasyon
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Modelin beklediği şekle sokma

    # **📊 Tahmin Al**
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], img_array)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction) * 100  # Güven skoru (%)

    # **🔍 Sonuçları Göster**
    if predicted_class in kategori_aciklamalari:
        st.success(f"🔍 **Tahmin Edilen Kategori:** {kategori_aciklamalari[predicted_class]}")
        st.info(f"🎯 **Güven Skoru:** %{confidence_score:.2f}")
        st.markdown(f"✍️ **Çizim Önerisi:** {cizim_onerileri[predicted_class]}")
    else:
        st.warning("⚠️ Model bir tahmin yapamadı, lütfen tekrar deneyin!")