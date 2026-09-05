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

---

## 10. YENİ MAKİNE KURULUMU (5 Eylül 2026)

Proje bu tarihten itibaren şu makinede yürüyor: i7-7700HQ (4C/8T, AVX2), 7,6 GB RAM,
native Ubuntu 24.04, GPU yok. Diskler: SSD `/` (sistem, kod, venv), NVMe `/mnt/nvme`
(hızlı geçici alan), HDD `/mnt/disk2tb` (NTFS, ntfs3 ile bağlı; checkpoint ve çalışma alanı).

| Ne | Nerede |
|---|---|
| Çalışan repo | `/home/ibox/calisma/LazyLora` (devir paketinden klon, git geçmişi dahil) |
| Referans motor | `/home/ibox/calisma/kimi-k3-in-c` (derlendi, `make test` 22/22 geçti) |
| Python ortamı | `/home/ibox/venvs/lazylora` (torch 2.14 CPU, numpy 2.5, safetensors, tiktoken) |
| Model | `/mnt/disk2tb/hamza/kimi_k3_model_weights` |
| Çalışma alanı | `/mnt/disk2tb/hamza/LazyLora_Workspace` (veri seti, önbellek, ölçüm günlükleri) |
| Aktivasyon / checkpoint | `/mnt/nvme/lazylora/{activations,checkpoints}` |
| Devir paketi (dokunma) | `/mnt/disk2tb/hamza/LazyLora_Handoff` |

Bütün yollar artık `lazy_lora/core/config.py` içindeki `default_*_dir()` fonksiyonlarından
gelir ve ortam değişkeniyle ezilebilir: `LAZYLORA_MODEL_DIR`, `LAZYLORA_WORKSPACE_DIR`,
`LAZYLORA_FAST_SCRATCH_DIR`, `LAZYLORA_ACTIVATION_DIR`, `LAZYLORA_CHECKPOINTS_DIR`,
`LAZYLORA_DATASET_DIR`, `LAZYLORA_CACHE_DIR`, `LAZYLORA_REF_FIXTURES`, `LAZYLORA_PYTHON`.

### Sentetik yedek politikası değişti

Bölüm 6.1'deki tuzak kapatıldı: bir tensör diskte yoksa motor artık **hata fırlatır**
(`MissingTensorError`). Rastgele ağırlıkla devam etmek yalnızca `LAZYLORA_ALLOW_SYNTHETIC=1`
ile mümkündür; bunu yalnızca `scripts/run_mock_tests.sh` ve `lazy_lora/tests/__init__.py`
ayarlar. NumPy yolundaki rastgele latent projeksiyonlar da (rapor K3) diskten okunur oldu.

### Checkpoint eksik: shard 68 ve 69

`model-00068` ve `model-00069` bu diskte **0 bayt** (26 Ağustos'tan beri). Katman 67 ve 68'in
tamamı eksik. Bölüm 2'deki "hepsi temiz" ifadesi yanlıştı; eski tarama boş shard'ı
"0 damaged" saymıştı. Yeni kontrol: `python scripts/check_shards.py` (ağsız, 5 saniye;
`train_lazy_lora.sh` bunu geçmeden başlamaz). İndirme: `LazyLora_Workspace/fetch_shards.py`
(kaldığı yerden devam eder, sha256 doğrular).

Ayrıca `workspace/c_ref13.log` dikkatle okunmalı: 13 katmanlı C referans koşusu katman 9'da
6 uzmanı okuyamayıp düşürmüş ve kendini `RUN INVALID` ilan etmiş. Katman 9'daki cosine
düşüşü (0,9966) muhtemelen bizim değil, referansın hatasıdır. Katman 0-8 karşılaştırması
geçerlidir; 9-12 yeniden üretilmelidir.

### Modelsiz doğrulama

```bash
export PYTHONPATH=/home/ibox/calisma/LazyLora
/home/ibox/venvs/lazylora/bin/python -m unittest lazy_lora.tests.test_reference_ops   # 8/8
bash scripts/run_mock_tests.sh                                                        # mock süiti
python scripts/check_shards.py                                                        # checkpoint bütünlüğü
python scripts/compare_with_c_dump.py --dump <chdump> --ids 19180,11 --layers 13      # gerçek ağırlık, C dökümüne karşı
```

## 11. GERİ GEÇİŞ YENİDEN YAZILDI VE DOĞRULANDI (5 Eylül 2026)

Rapordaki K1, K2, K8 maddeleri kapatıldı.

**Ne değişti (`lazy_lora/trainer/lazy_trainer.py`):**
- Katmanın hesabı tek bir yerde: `_run_layer(layer_idx, h_in, bank)`. İleri geçiş bunu
  `no_grad` altında, geri geçiş `enable_grad` altında **aynı kodu tekrar oynatarak** çalıştırır.
  Banka karışımı, blok sınırında akışın sıfırlanması, latent RMSNorm jakobiyeni, yönlendirici
  ağırlıklarından akışa dönen gradyan: hepsi autograd'dan gelir, elle türetilmiş kısım yok.
- Yönlendirilmiş uzmanlar tek bir `torch.autograd.Function` (`RoutedExpertsFunction`): ileri
  geçişte uzmanlar bir kez, geri geçişte bir kez daha diskten akıtılır; LoRA gradyanları ve
  dL/dh_latent orada elle hesaplanır (600 uzmanı autograd grafiğinde tutmak onlarca GB olurdu).
- Banka gradyanı: banka girdileri sınır katmanlarının (0, 12, 24, ...) h_in'idir ve zaten
  aktivasyon tamponunda durur; geri geçiş bankayı oradan yeniden kurar. Bankaya akan gradyan
  `_grad_bank`'ta bekletilir ve sıra o sınır katmanına gelince grad_h_in'e eklenir.
- Çıkış karışımı + son norm da (`_finalize_backward`) artık türevleniyor; eskiden atlanıyordu.
- Uzman toplamı fp32'de birikiyor (C referansı gibi); bf16 birikim katman çıktısını ~1e-3
  kaydırıp birkaç katman sonra yönlendirmeyi değiştiriyordu.
- LoRA parametreleri fp32 (K2). `lora_dropout` 0 (yeniden hesaplamalı geri geçişle uyumsuz).
- NumPy yolu kaldırıldı (dikkat zaten torch istiyordu; ölü koddu ve K3'ün kaynağıydı).

**Doğrulama (`scripts/verify_backward.py`, gerçek ağırlıklar, fp32, 4 token):**
Analitik yönlü türev ile merkezi sonlu fark, LoRA tensörleri + h_in + banka yönlerinde:

| Katman | Özellik | En kötü göreli hata |
|---|---|---|
| 1 | KDA + MoE, 1 banka girdisi | 1.1e-3 |
| 12 | blok sınırı (banka itme, akış sıfırlama) | 2.0e-3 |
| 13 | 2 banka girdisi | 9.1e-4 |
| 1 | 16 LoRA tensörünün tamamı (q/v dikkat dahil) + h_in + banka | 9.1e-3 (türevi ~3e-4 olan 2 yön; kalanı ≤2e-3) |
| 3 | MLA, 16 tensörün tamamı + h_in + banka | 3.6e-3 |

Günlükler: `LazyLora_Workspace/k8_layer*_2026-09-05.log`. İleri geçiş değişmedi
(`cmp13_after_refactor2_2026-09-05.log`, önceki koşuyla birebir).

**Ölçülen maliyet (bu makine, fp32, 4 token):** katman başına geri geçiş 37-46 s.

**Not:** Sonlu fark fp32 motorda bile ancak kayıp float64'te toplanıp adım yöne göre
ölçeklenince anlamlı çıktı; ilk sürüm (fp32 kayıp, sabit eps=1e-3) 1-2 ulp'lik farklar ölçüp
sahte uyumsuzluk raporladı. Harness'ı değiştirirken buna dikkat.

## 12. TAM CHECKPOINT (K4, 5 Eylül 2026)

`save_lora_checkpoint(step, data_cursor)` artık LoRA tensörleriyle birlikte Adam momentlerini,
adım sayacını, LR zamanlayıcısını, torch/numpy RNG durumlarını ve veri kümesi imlecini tek
dosyaya yazar; önce `.tmp`'ye yazılıp `fsync` sonrası `os.replace` ile yerine konur (yarım
dosya asla nihai adı taşımaz). `checkpoints/latest.txt` son dosyanın adını tutar.

Devam etmek: `python -m lazy_lora.trainer.lazy_trainer --resume <ckpt.pt> --steps N`
(`LazyLoRATrainer.load_checkpoint`, `train(resume_from=...)`; veri imleci
`StreamingDatasetIterator.get_batches(skip_samples=...)` ile uygulanır).
Eski biçim (yalnızca LoRA sözlüğü) da yüklenir, ama optimizer sıfırdan başlar.

## 13. TEMİZ C REFERANSI VE ÖLÇÜM ALETİ (5 Eylül 2026, akşam)

C motoru bu makinede 13 katmanla yeniden koşturuldu (`LazyLora_Workspace/run_cref13.sh`).
Boş shard'lar yüzünden motor tam dizini reddettiği için katman 0-12 + embed/lm_head
shard'larına sembolik bağlarla `/mnt/nvme/lazylora/k3_partial13` dizini kuruldu. Koşu geçerli
(uzman düşürme yok, exit 0), tepe RSS 6.11 GB, 169 s. Yeni döküm: `chdump_2026-09-05/`.

Yeni dökümle karşılaştırma (`cmp13_vs_newdump_2026-09-05.log`): katman 9 kosinüs
0.9966 → **0.99951**, katman 12 → **0.99995**. Eski dökümdeki düşüş referansın düşürdüğü
6 uzmandandı; motorumuzda katman 0-12'de yapısal fark yok.

**Uzman erişim izi** (`lazy_lora/monitor/trace.py`): `trainer.trace` ayarlıysa her katmanın
(token, seçilen 16 uzman, ağırlık) kaydı `trace.bin`'e, süre/bayt/tepe RSS `trace.json`'a
yazılır. `scripts/measure_routing.py` bir metni bu izle ileri geçirir;
`scripts/analyze_trace.py` yoğunlaşma, entropi, etkin uzman sayısı, zamansal ve katmanlar
arası Jaccard, N'ye göre benzersiz uzman eğrisi (tek koşudan, nedensellik sayesinde) ve iki
iz arasında dil karşılaştırması üretir. İstemler `LazyLora_Workspace/prompts/` (aynı anlam:
Türkçe 263 token, İngilizce 158 token; Türkçe %66 daha fazla token harcıyor).
