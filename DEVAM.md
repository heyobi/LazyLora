# 🔄 DEVİR BELGESİ — LazyLoRA'nın Yeni Makinede Kaldığı Yerden Sürdürülmesi

Bu belge, projeyi başka bir bilgisayarda devralacak kişinin (veya oturumun) sohbet geçmişi olmadan devam edebilmesi için yazılmıştır. Tarih: **29 Ağustos 2026**.

---

## 1. TEK CÜMLEYLE DURUM

LazyLoRA motoru, **2,78 trilyon parametreli Kimi K3'ün ileri geçişini bağımsız bir referans implementasyona karşı doğrulanmış biçimde yeniden üretiyor**; geri geçiş ve optimizer adımı çalışıyor; eğitilmiş bir adaptör henüz yok.

---

## 2. NEREDE KALDIK

### ✅ Tamamlananlar

| İş | Kanıt |
|---|---|
| Gerçek KDA + MLA dikkat katmanları | Referans fixture'larıyla 1e-5 toleransında eşleşme |
| MXFP4 uzman çözme | Kuantize olmayan paylaşılan uzmanla karşılaştırma: absmean 0,0194 ↔ 0,0149 |
| Katman başına gerçek yönlendirici ağırlıkları | Diskten okunuyor (önce rastgele üretiliyordu) |
| SiTU aktivasyonu | `4·tanh(g/4)·σ(g) × 25·tanh(u/25)` — fixture ile doğrulandı |
| Blok-artık mekanizması | Katman 12'de akış sıfırlanıyor, C motoruyla birebir |
| **13 katman uçtan uca doğrulama** | **cosine 0,9989 – 0,99999** (bkz. Bulgular.md §15) |
| Okuma yolu optimizasyonu | `mmap` → `pread`: 21,8 → 101 MB/s |
| MXFP4 çözücü optimizasyonu | 9,04 s → 2,53 s (12 uzman) |
| Bozuk veriye dayanıklılık | Ölçek baytı ≥253 olan grup sıfır katkı |
| Checkpoint bütünlüğü | 92 katman tarandı; hasarlı 3 parça yeniden indirildi, **hepsi temiz** |

### ⏳ Yapılmayanlar

1. **Öğrenme kanıtı koşusu** — 5-10 örnek, 15-20 adım, düşen loss eğrisi. Sıradaki iş budur.
2. **Fikir 9** (Fikirler.md) — geri geçişte uzmanları tekrar okumamak; adım süresini ~yarıya indirir.
3. **Uzman başına token toplama (gather)** — şu an her aktif uzman *tüm* token'lar üzerinde çalıştırılıp maskeleniyor. 24 token'da sorun değil, 512 token'da 56 kat fazla hesap demek. Uzun dizilerle çalışmanın ön koşulu.
4. `gradient_accumulation_steps` config'de tanımlı ama **kodda kullanılmıyor**; her `train_step` bir optimizer adımı atıyor.

---

## 3. YENİ MAKİNEDE İLK KURULUM

```bash
# 1. Depoları yerleştir
#    LazyLora/            -> proje kökü
#    kimi-k3-in-c/        -> referans motor (teşhis yaması dahil)

# 2. Sanal ortam (venv KOPYALANMAZ, yeniden kurulur)
python3 -m venv /path/to/workspace/venv
/path/to/workspace/venv/bin/pip install -r requirements.txt
#    Not: requirements.txt bu makinenin tam donmuş listesidir. Gerekli asgari set:
#    torch (CPU sürümü yeterli), numpy, transformers, tokenizers, tiktoken,
#    safetensors, huggingface_hub, psutil, rich

# 3. Yolları güncelle — lazy_lora/core/config.py içindeki PathConfig
#    base_model_dir  -> Kimi K3 ağırlıklarının yeni konumu
#    workspace_dir   -> aktivasyon tamponu, checkpoint, veri seti için yazılabilir dizin
```

**Referans fixture'ların yolu** [lazy_lora/tests/test_reference_ops.py](lazy_lora/tests/test_reference_ops.py) içinde sabit yazılıdır (`FIXTURES`); yeni konuma göre güncellenmelidir. Dizin yoksa testler atlanır, hata vermez.

### Doğrulama (kurulumdan sonra ilk iş)

```bash
export PYTHONPATH=/path/to/LazyLora
venv/bin/python -m unittest lazy_lora.tests.test_reference_ops   # 8 test, saniyeler sürer
venv/bin/python -m unittest discover -s lazy_lora/tests -t .     # tam süit, ~3 dk
```

Referans op testleri geçiyorsa motorun matematiği sağlamdır.

---

## 4. SIRADAKİ ADIM: ÖĞRENME KANITI KOŞUSU

**Amaç:** Küçük bir sette loss'un düştüğünü göstermek. Kasıtlı ezberleme — genelleme değil, mekanizmanın çalıştığının kanıtı.

**Ayarlar** (`lazy_lora/core/config.py` veya CLI):
- `warmup_steps = 2` — varsayılan 50, 20 adımlık koşuda öğrenme oranını bastırır
- `learning_rate = 1e-3` civarı — küçük seti ezberletmek istiyoruz
- 5-10 örnek, `--seq-len 24`, 15-20 adım

```bash
bash scripts/train_lazy_lora.sh --steps 20 --seq-len 24 --lr 1e-3
```

Loss her adımda `workspace/forward_loss.jsonl` dosyasına yazılır (ileri geçiş biter bitmez, geri geçişi beklemeden). Grafik o dosyadan çizilir.

**Beklenen:** İlk adımda 2-5 arası bir loss, sonraki adımlarda düşüş. Loss ~12 (=ln 163840) ise boru hattı bozuktur.

---

## 5. ÖLÇÜLMÜŞ MALİYETLER (referans için)

Mekanik disk (WD20EZBX, ~100 MB/s), 24 token, `pread` ve hızlı çözücü sonrası:

| | Değer |
|---|---|
| Katman başına | ~112 s |
| İleri geçiş (93 katman) | ~2,9 saat |
| Geri geçiş | ~3,5 saat (Fikir 9 ile ~1 saat) |
| Katman başına okuma | 4-6 GB (aktif uzman sayısına göre) |

**SSD'ye taşındığında** (SATA ~500 MB/s) bu süreler 5 kat kısalmalıdır: adım ~35-40 dakika.

> **Uyarı:** Bu ölçümler WSL2/DrvFs üzerindedir. Native Linux + ext4'te okuma yolu daha da hızlanır; `mmap` yerine `pread` seçimi DrvFs'e özgü bir kazançtı, ext4'te ikisi de hızlıdır.

---

## 6. BİLİNEN TUZAKLAR

1. **Sessiz sentetik yedekler.** Motorun her yükleyicisi, tensör bulunamazsa rastgele ağırlık üretir. Bu, projedeki en pahalı hataların kaynağıydı (bkz. Bulgular.md §11, §15.4). Bir tensör adı değişirse kod çalışmaya devam eder ama **anlamsız sonuç üretir**. Yeni bir yol/isim eklerken mutlaka gerçek ağırlıkla doğrula.

2. **Mock testler gerçek modeli görmemeli.** `base_model_dir` boş bir dizine ayarlanmazsa testler gerçek 7168-boyutlu tensörleri okur ve hiçbir şey doğrulamaz.

3. **Loss dosyası paylaşılır.** Mock testler de `workspace/forward_loss.jsonl` dosyasına yazar; gerçek koşu loss'unu okurken karıştırma.

4. **Checkpoint hasarı sessizdir.** Bozuk MXFP4 ölçekleri `inf` üretip tüm ileri geçişi NaN'e çevirebilir. Koruma eklendi ama bir tarama aracı da var:
   ```bash
   # Ölçek baytları sağlıklı modelde 110-125 arasında kümelenir; std > 40 hasar demektir
   ```
   Tam tarama betiği git geçmişinde (`scan_tmp.py`, commit mesajlarında anlatılıyor).

---

## 7. REFERANS MOTOR (kimi-k3-in-c)

Doğrulamanın temel aracı. İki kullanım:

```bash
# Katman katman gizli durum dökümü (teşhis yaması, yalnızca ortam değişkeni verilince çalışır)
K3_DUMP_H=/tmp/dump ./bin/k3 <model_dir> --ids 19180,11 --layers 4 --gen 1 \
    --trunk <trunk_dir> --trunk-gb 2.5 --cache-gb 0.35 --dump-logits /tmp/logits.bin
```

Bizim motorumuzun çıktısı bununla karşılaştırılır. LoRA'nın $B$ matrisi sıfırla başladığı için **eğitilmemiş adaptörlerle logits birebir aynı olmalıdır**; fark varsa ileri geçişte hata var demektir.

Op-bazlı fixture'lar: `kimi-k3-in-c/tests/fixtures/ops/` — kendi ağırlıkları, girdileri ve beklenen çıktılarıyla. Bunlar [test_reference_ops.py](lazy_lora/tests/test_reference_ops.py) ile koşuluyor.

---

## 8. OKUMA SIRASI

1. **[Bulgular.md](Bulgular.md)** — özellikle §10-15: donanım gerçeği, mimari düzeltmeleri, referans doğrulama
2. **[Fikirler.md](Fikirler.md)** — 9 fikir; 8 ve 9 hız üzerine, 6 ve 7 eğitim stratejisi üzerine
3. **`git log`** — her commit mesajı neyin neden değiştiğini ve hangi ölçümün buna yol açtığını anlatır

---

## 9. DÜRÜST DEĞERLENDİRME

**Başarılan:** 1,45 TB'lık bir modelin 8 GB RAM'de doğru şekilde çalıştırılması ve LoRA adaptörleriyle eğitilebilir hale getirilmesi. Referansa karşı ölçülmüş.

**Başarılamayan:** Eğitilmiş bir Türkçe adaptör. Sebep hesap değil, **veri hacmi**: adım başına 1,45 TB'ın diskten geçmesi gerekiyor.

**Ölçek gerçeği:** LIMA benzeri 1000 örneklik bir set, 512 token'lık dizilerle mekanik diskte ~2 yıl, SATA SSD'de ~4-5 ay sürer. Anlamlı hedef, birkaç yüz adımlık odaklı bir eğitimdir.
