## 🧠 TrVLM Nedir?
TrVLM, SigLIP görsel kodlayıcısı ile GPT2-large Türkçe dil modelini birleştiren çok modlu (multimodal) büyük bir dil modelidir. Görselleri anlamlandırarak doğal Türkçe metin üretimi yapabilen bu model, hem görsel açıklama hem de görsel soru-cevap görevlerini yerine getirebilecek şekilde tasarlanmıştır.

## 🔧 Geliştirme Süreci
1️⃣ Tek Modalite Ön Eğitimi
Modelin her bileşeni sıfırdan eğitilmemiş, yerine güçlü ön-eğitimli modeller entegre edilmiştir:

Görsel kodlayıcı: google/siglip-base-patch16-224

Dil modeli: ytu-cosmos/Turkish-Llama-8b-Instruct-v0.1

2️⃣ Özellik Uyarlama
LLaVA eğitim yaklaşımı izlenerek, sadece görsel projektör katmanı 500K görsel-metin çifti ile eğitilmiştir. Bu sayede görsel ve dil temsilleri birleştirilerek uyumlu hale getirilmiştir.

Huggingface Veriseti : emrecn/Predataset

3️⃣ Görev Spesifik Eğitim
Uyumlaştırılan model, üç temel görev için ayrı ayrı eğitilmiştir:

Kısa Açıklama

Detaylı Açıklama

Görsel Soru-Cevaplama (VQA)

Eğitim için, 1 milyondan fazla resim-soru-cevap üçlüsünü içeren bir veri kümesi kullanılmıştır.

Huggingface Veri setleri :

Kısa : emrecn/newveri ,

Uzun : emrecn/longdataset

Vqa : berkanbucak/finetunevqa

---

## 🧩 Model Details

### Model Description

- **Developed by:** Emre Ceniklioğlu, Berkan Bucak, Cihan Deniz Ekiz
- **Affiliation:** Ankara University - Computer Engineering Department  
- **Model Type:** Multi-modal Encoder-Decoder (Image-Text-to-Text), Causal Language Model  
- **Language(s):** Turkish (`tr`)
---

## 🧩 Kullanım Alanları – TrVLM
Aşağıda, TrVLM çok modlu görsel-dil modelinin doğrudan, dolaylı ve alan dışı kullanım senaryoları açıklanmıştır.

### ✅ Doğrudan Kullanım Alanları
🔹 Kısa Açıklama (Short Captioning)
Bu görevde TrVLM, görselin hızlı ve özlü bir açıklamasını yapar. Kullanıcılar modele şu tür prompt'lar verebilir:

"Kısaca açıkla", "Görseli özetle", "Çok kısa özetle"

Model, görselin genel içeriğini kısa bir şekilde ifade eder.
Not: Bu görev için model, düşük halüsinasyon eğilimindedir ve yüksek doğrulukla çalışır. Farklı üretim parametreleriyle çıktılarınızı özelleştirebilirsiniz.

🔹 Detaylı Açıklama (Detailed Captioning)
Bu görevde TrVLM, görseldeki tüm detayları daha kapsamlı şekilde açıklamaya çalışır. Aşağıdaki gibi prompt’lar verilebilir:

"Detaylı açıkla", "Görseli çok detaylı anlat"

Model genellikle başarılı cevaplar üretir ancak zaman zaman görselde olmayan detaylar hayal edebilir (halüsinasyon).
Not: Bu görevin çıktılarında dikkatli değerlendirme önerilir.

🔹 Görsel Soru-Cevaplama (Visual Question Answering)
Model, verilen bir görsele ilişkin sorulara yanıt verir. Örnek prompt’lar:

"Bu görselde kaç kişi var?", "Adam ne yapıyor?", "Bu araba ne renk?"

Not: Bu görevde model başarılı olsa da bazen görselde olmayan cevaplar üretme eğilimindedir. Üretim ayarları ile çıktılar kontrol altına alınabilir.

### 🚫 Alan Dışı Kullanım Senaryoları
Aşağıdaki durumlar için model uygun değildir:

Çok turlu diyaloglar / chat senaryoları: TrVLM geçmiş bilgiyi tutmaz. Çok turlu konuşmalar için eğitilmemiştir.

Çoklu görsel karşılaştırma: Aynı anda birden fazla görsel girişi desteklemez.

OCR, Segmentasyon, Çoklu Obje Tanıma: Bu görevler için model eğitilmemiştir. 

---

## ⚠️ Bias, Risks, and Limitations

Model, internetten toplanmış Türkçe metin ve görsellerle eğitilmiştir. Dolayısıyla:

- Önyargılı, toplumsal hassasiyet içeren ifadeler üretme riski taşır.
- Görsel içeriği yanlış yorumlayabilir.
- Gerçeklikten uzak yanıtlar verebilir.
---

## 🚀 How to Get Started with the Model

TrVLM modelini denemenin en kolay yolu, tüm gerekli dosyaları indirip inference.py dosyasını çalıştırmaktır. Bu dosya, Gradio tabanlı bir web arayüzü başlatarak, kullanıcıların modele kolayca görsel ve metin girişi yaparak çıktıları test etmesini sağlar.

🔧 Kurulum ve Çalıştırma
Gerekli kütüphaneleri yükleyin:

Aşağıdaki bileşenleri indirmeniz gerekmektedir:

base model ağırlıkları 

fine-tuned LoRA adapter ağırlıkları

💾 Tüm bu dosyalar aşağıdaki Google Drive bağlantısında mevcuttur:

Ardından !python inference.py komutu ile inference ortamını başlatabilirsiniz:

## 🏋️ Training Details

⚙️ Training Procedure

Pretrain: Sadece görselden metne projeksiyon katmanı eğitildi

Başarı: Model, görsel-metin eşleştirmede temel yetkinlik kazandı.

| Özellik               | Değer                    |
| --------------------- | ------------------------ |
| **Veri Sayısı**       | 575K görüntü-metin çifti |
| **Global Batch Size** | 16                      |
| **Learning Rate**     | 2e-4                   |
| **Epochs**            | 2                        |
| **Max Length**        | 256                     |
| **Weight Decay**      | 0.1                     |

Donanım: 1x NVIDIA A100 GPU (Google Colab), 4 saat süre.

Fine-tune: 1. Çok Görevli Fine-Tuning (Multitask Fine-Tuning)

Amaç: Modelin birden fazla görevde (VQA, kısa/uzun açıklama üretme) aynı anda performansını artırmak.

Dondurulan Bileşenler: SIGLIP encoder ve Turkish LLaMA'nın çoğu katmanı (sadece projeksiyon katmanı ve dil modelinin önemli katmanları eğitildi).

LoRA (Low-Rank Adaptation): Dil modelinin dikkat katmanlarına hafif adaptasyon uygulandı.

| Özellik               | Değer                                   |
| --------------------- | --------------------------------------- |
| **Veri Sayısı**       | 1.1M görsel + talimat + cevap üçlüsü    |
| **Global Batch Size** | 8                                       |
| **Learning Rate**     | 5e-5                                    |
| **Epochs**            | 1                                       |
| **Max Length**        | 256                                     |
| **Weight Decay**      | 1e-6                                    |

2. Kısa Açıklama için Fine-Tuning
   
Amaç : Multitask modelin kısa açıklamalardaki başarısını daha da artırmak.

Strateji: Tam fine-tuning yerine LoRA kullanıldı.

Adapte Edilen Katmanlar: Dil modelinin dikkat mekanizmaları ve projeksiyon katmanı.

Avantaj: Modelin kısa açıklama üretme yeteneğini arttırmak.

| Özellik               | Değer                                   |
| --------------------- | --------------------------------------- |
| **Veri Sayısı**       | 510k kısa açıklama veri seti |
| **Global Batch Size** | 8                                       |
| **Learning Rate**     | 1e-5                                    |
| **Epochs**            | 3                                       |


🖥️ Compute Infrastructure
GPU: NVIDIA A100 (Colab Pro+)

Frameworks: PyTorch, Hugging Face Transformers

Transformers

📊 Evaluation

### **1. Metrik Tabanlı Değerlendirme**
Model, 4 farklı görev türünde (Multitask, Uzun Açıklama, VQA, Kısa Açıklama) 8 metrikle test edilmiştir. Kullanılan metrikler:

#### **Performans Tablosu**:
| **Metrik**           | **Multitask** | **Uzun Açıklama** | **VQA**      | **Kısa Açıklama** |
|----------------------|---------------|-------------------|--------------|-------------------|
| **ROUGE-1**          | 0.2494        | 0.3177            | 0.3533       | **0.4816**        |
| **ROUGE-2**          | 0.1311        | 0.0907            | 0.1842       | **0.2835**        |
| **ROUGE-L**          | 0.2049        | 0.1901            | 0.3074       | **0.4164**        |
| **BLEU**             | 0.0352        | 0.0063            | 0.0555       | **0.0976**        |
| **METEOR**           | 0.1461        | 0.0824            | 0.2174       | **0.2921**        |
| **BERT Precision**   | 0.5546        | 0.5252            | 0.6676       | **0.6824**        |
| **BERT Recall**      | 0.1461        | **0.5771**        | 0.6335       | 0.6575            |
| **BERT F1**          | 0.5110        | 0.5493            | 0.6462       | **0.6687**        |


---

### **2. GPT-4o Hakem Değerlendirmesi**
100 örnek üzerinde GPT-4o'nun insan benzeri değerlendirmesi (0-10 arası puanlama):

| **Kriter**                     | **Puan** |
|-------------------------------|----------|
| **Talimat Uyumu**             | 6.25     | 
| **Görsel-Metin Tutarlılığı**  | 7.30     | 
| **Anlamsal Doğruluk**         | 6.80     | 
| **Yaratıcılık & Çeşitlilik**  | 5.50     | 
| **GENEL ORTALAMA**            | 6.46     |

---

### **3. Kritik Sonuçlar ve Öneriler**
#### **Başarılar**:
1. ✅ **Kısa Açıklamada Sınıf Lideri**: Tüm metriklerde en yüksek skor.
2. ✅ **Görsel-Metin Tutarlılığı**: GPT-4o ile 7.3 puan (projenin ana gücü).
3. ✅ **Düşük Kaynak Verimliliği**: 1x A100 GPU ile 4 saatlik pretrain.

#### **Performans Özet Tablosu**:
| **Görev**         | **Güçlü Yönler**       | **Zayıf Yönler**         | **İyileştirme Önceliği** |
|-------------------|------------------------|--------------------------|--------------------------|
| Kısa Açıklama     | ROUGE, BERTScore       | -                        | ⭐ (En iyi)              |
| VQA               | Anlamsal doğruluk      | Kelime seçimi hataları   | ⭐⭐                      |
| Uzun Açıklama     | BERT Recall            | Akıcılık ve zenginlik    | ⭐⭐⭐ (Acil)              |
| Multitask         | Denge                  | Görev çakışması          | ⭐⭐                      |

**Sonuç**: Model, Türkçe VLM'ler arasında **kısa açıklama ve VQA'da başarılı**, ancak uzun metin üretimi ve yaratıcılıkta gelişime açık.

🧠 Model Architecture and Objective

Görsel encoder: SigLIP (Google)

Projeksiyon katmanı: MLP (768 → 4096)

LLM: Turkish LLaMA 8B

🙋 Model Card Contact

Emre Ceniklioğlu – GitHub 

Mail: emreceniklioglu11@gmail.com


