import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# Uyarıları kapat
warnings.filterwarnings('ignore')

# ==============================================================================
# 🎛️ WEB TASARIM VE PANEL AYARLARI
# ==============================================================================
st.set_page_config(
    page_title="AI İddaa Global Analyzer v7.0",
    page_icon="🔮",
    layout="wide"
)

# Özelleştirilmiş Şık CSS Tasarımları (Açık, Yumuşak ve Okunaklı Tonlar)
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        .header-panel {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        h1, h2, h3 { color: #212529 !important; font-weight: 700; }
        hr { border: 0; height: 1px; background: #dee2e6; margin: 25px 0; }
    </style>
""", unsafe_allow_html=True)

# Başlık Bölümü Paneli
st.markdown("""
    <div class="header-panel">
        <h1 style='color: white !important; margin: 0;'>🔮 Ultra Yapay Zeka Sürpriz & Evrensel Analiz Merkezi</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>
            <b>Sürüm 7.0 Profesyonel Güncelleme:</b> Kelly Kriteri Kasa Yönetimi, Value Bet Dedektörü ve Dinamik Lig Filtreleme Entegre Edildi.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 📊 1. ADIM: VERİ OKUMA VE SÜRPRİZ ENDEKSİ HESAPLAMA
# ==============================================================================
ihtiyacimiz_olan_sutunlar = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A'
]

try:
    # Veri setini yükle ve temizle
    df_gecmis = pd.read_csv('maclar.csv', usecols=ihtiyacimiz_olan_sutunlar)
    df_gecmis = df_gecmis.dropna()

    # Tarih formatlarını düzelt ve sırala
    df_gecmis['Date'] = pd.to_datetime(
        df_gecmis['Date'],
        dayfirst=True,
        errors='coerce',
        format='mixed'
    )
    df_gecmis = df_gecmis.sort_values(by='Date').reset_index(drop=True)

    # Zaman ağırlıklı yapay zeka hafıza katsayısı (Exp Weighting)
    toplam_mac = len(df_gecmis)
    df_gecmis['Ağırlık'] = np.exp(6 * (df_gecmis.index / toplam_mac)) / np.exp(6)

except FileNotFoundError:
    st.error(
        "❌ **'maclar.csv' dosyası bulunamadı!** "
        "Lütfen dosya adının ve uzantısının doğru olduğundan emin olun."
    )
    st.stop()

# ==============================================================================
# 📈 DİNAMİK LİG SÜRPRİZ VE BOL GOL ENDEKSLERİ
# ==============================================================================
lig_surpriz_katsayilari = {}

for lig in df_gecmis['Div'].unique():
    df_lig = df_gecmis[df_gecmis['Div'] == lig]
    toplam_lig_maci = len(df_lig)

    if toplam_lig_maci > 0:
        alti_arti_oran = len(df_lig[(df_lig['FTHG'] + df_lig['FTAG']) >= 6]) / toplam_lig_maci
        lig_surpriz_katsayilari[lig] = {
            'alti_arti_egilimi': max(alti_arti_oran, 0.01) * 10,
            'kaos_faktoru': 1.0 + (alti_arti_oran * 2)
        }
    else:
        lig_surpriz_katsayilari[lig] = {'alti_arti_egilimi': 0.1, 'kaos_faktoru': 1.0}


def agirlikli_lig_ortalamalari(df):
    """Liglerin zaman ağırlıklı ev sahibi ve deplasman gol ortalamalarını hesaplar."""
    sozluk = {}
    for lig in df['Div'].unique():
        df_lig = df[df['Div'] == lig]
        toplam_w = df_lig['Ağırlık'].sum()

        ev_ort = (df_lig['FTHG'] * df_lig['Ağırlık']).sum() / toplam_w
        dep_ort = (df_lig['FTAG'] * df_lig['Ağırlık']).sum() / toplam_w
        sozluk[lig] = {'ev_ort': ev_ort, 'dep_ort': dep_ort}
    return sozluk


lig_istatistikleri = agirlikli_lig_ortalamalari(df_gecmis)


def takim_elite_ratingleri(df, lig_istatistikleri):
    """Takımların ofansif ve defansif güç ratinglerini hesaplar."""
    ratingler = {}
    for lig, ort in lig_istatistikleri.items():
        df_lig = df[df['Div'] == lig]
        takimlar = set(df_lig['HomeTeam'].unique()).union(set(df_lig['AwayTeam'].unique()))

        for takim in takimlar:
            df_ev = df_lig[df_lig['HomeTeam'] == takim]
            df_dep = df_lig[df_lig['AwayTeam'] == takim]

            # İç saha performans ratingi
            if not df_ev.empty:
                w_ev = df_ev['Ağırlık'].sum()
                hucum_ic = ((df_ev['FTHG'] * df_ev['Ağırlık']).sum() / w_ev) / ort['ev_ort']
                savunma_ic = ((df_ev['FTAG'] * df_ev['Ağırlık']).sum() / w_ev) / ort['dep_ort']
            else:
                hucum_ic, savunma_ic = 1.0, 1.0

            # Dış saha performans ratingi
            if not df_dep.empty:
                w_dep = df_dep['Ağırlık'].sum()
                hucum_dis = ((df_dep['FTAG'] * df_dep['Ağırlık']).sum() / w_dep) / ort['dep_ort']
                savunma_dis = ((df_dep['FTHG'] * df_dep['Ağırlık']).sum() / w_dep) / ort['ev_ort']
            else:
                hucum_dis, savunma_dis = 1.0, 1.0

            ratingler[(lig, takim)] = {
                'hucum_ic': hucum_ic, 'savunma_ic': savunma_ic,
                'hucum_dis': hucum_dis, 'savunma_dis': savunma_dis
            }
    return ratingler


takim_veritabani = takim_elite_ratingleri(df_gecmis, lig_istatistikleri)

# 📊 Üst Bilgi Metrik Kartları
col1, col2 = st.columns(2)
col1.metric(
    label="📚 Yapay Zekanın Hafızası",
    value=f"{len(df_gecmis):,} Maç",
    help="Modellenen ve incelenen toplam geçmiş maç sayısı"
)

try:
    df_ham = pd.read_csv('guncel.csv')
    for col in df_ham.columns:
        if col.lower() == 'date':
            df_ham = df_ham.rename(columns={col: 'Date'})

    df_ham = df_ham.dropna(subset=['Div', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A'])
    col2.metric(
        label="🔮 Analiz Edilecek Güncel Bülten",
        value=f"{len(df_ham)} Maç",
        help="guncel.csv dosyasından okunan ve işlenecek aktif maç sayısı"
    )
except FileNotFoundError:
    df_ham = pd.DataFrame()
    col2.metric(label="🔮 Analiz Edilecek Güncel Bülten", value="0 Maç (guncel.csv eksik)")

st.markdown("---")


# ==============================================================================
# 🧠 SÜRPRİZ DEDEKTÖRLÜ GELİŞMİŞ TAHMİN MOTORU
# ==============================================================================
def ultra_surpriz_tahmin_motoru(lig, ev, dep, ev_orani, ber_orani, dep_orani, tarih_metni):
    lig, ev, dep = str(lig), str(ev), str(dep)
    ev_orani, ber_orani, dep_orani = float(ev_orani), float(ber_orani), float(dep_orani)

    ort = lig_istatistikleri.get(lig, {'ev_ort': 1.35, 'dep_ort': 1.05})
    ev_r = takim_veritabani.get((lig, ev), {'hucum_ic': 1.0, 'savunma_ic': 1.0, 'hucum_dis': 1.0, 'savunma_dis': 1.0})
    dep_r = takim_veritabani.get((lig, dep), {'hucum_ic': 1.0, 'savunma_ic': 1.0, 'hucum_dis': 1.0, 'savunma_dis': 1.0})
    lig_kaos = lig_surpriz_katsayilari.get(lig, {'alti_arti_egilimi': 0.1, 'kaos_faktoru': 1.05})

    # Temel Gol Beklentileri (xG)
    ev_xg = ev_r['hucum_ic'] * dep_r['savunma_dis'] * ort['ev_ort']
    dep_xg = dep_r['hucum_dis'] * ev_r['savunma_ic'] * ort['dep_ort']

    # Bahis Şirketi Oran Gücü Dengesi
    toplam_oran_gucu = (1.0 / ev_orani) + (1.0 / ber_orani) + (1.0 / dep_orani)
    ev_piyasa = ((1.0 / ev_orani) / toplam_oran_gucu) * (ort['ev_ort'] + ort['dep_ort']) * 0.55
    dep_piyasa = ((1.0 / dep_orani) / toplam_oran_gucu) * (ort['ev_ort'] + ort['dep_ort']) * 0.45

    # Final Melez xG Hesaplaması
    ev_final_xg = float(max((ev_xg * 0.80) + (ev_piyasa * 0.20), 0.15))
    dep_final_xg = float(max((dep_xg * 0.80) + (dep_piyasa * 0.20), 0.15))

    p_ev, p_ber, p_dep, p_ust, p_kg_var, p_alti_arti = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def dixon_coles_ayari(x, y, mu, lam, rho=-0.05):
        """Düşük skorlu maçlardaki bağımlılığı düzelten Dixon-Coles kalibrasyonu."""
        if x == 0 and y == 0: return 1 - (mu * lam * rho)
        if x == 1 and y == 0: return 1 + (mu * rho)
        if x == 0 and y == 1: return 1 + (lam * rho)
        if x == 1 and y == 1: return 1 - rho
        return 1.0

    for x in range(10):
        for range_y in range(10):
            skor_p = float(poisson.pmf(x, ev_final_xg) * poisson.pmf(range_y, dep_final_xg))
            skor_p *= dixon_coles_ayari(x, range_y, ev_final_xg, dep_final_xg)

            if x > range_y:
                p_ev += skor_p
            elif x == range_y:
                p_ber += skor_p
            else:
                p_dep += skor_p

            if (x + range_y) > 2.5:
                p_ust += skor_p
            if x >= 1 and range_y >= 1:
                p_kg_var += skor_p
            if (x + range_y) >= 6:
                p_alti_arti += skor_p
    p_alti_arti_final = p_alti_arti * lig_kaos['kaos_faktoru']
    olasılıklar = [p_ev, p_ber, p_dep]
    en_büyük = max(olasılıklar)

    # --- ANA TAHMİN VE SKOR MODELLEMESİ ---
    if en_büyük == p_ev:
        ana_tahmin = "Ev Sahibi Kazanır (1)"
        secilen_olasilik = p_ev
        secilen_oran = ev_orani
        rozet = "💎 HAFTANIN BANKOSU" if (p_ev > 0.65 and ev_orani <= 1.85) else (
            "🔥 GÜVENLİ BANKO" if p_ev > 0.64 else "⭐ EV SAHİBİ ÖNDE"
        )
        en_iyi_ev_skor = int(np.ceil(ev_final_xg)) if ev_final_xg > dep_final_xg else int(np.floor(ev_final_xg)) + 1
        en_iyi_dep_skor = int(np.floor(dep_final_xg))
        if en_iyi_ev_skor <= en_iyi_dep_skor:
            en_iyi_ev_skor = en_iyi_dep_skor + 1

    elif en_büyük == p_dep:
        ana_tahmin = "Deplasman Kazanır (2)"
        secilen_olasilik = p_dep
        secilen_oran = dep_orani
        rozet = "💎 HAFTANIN BANKOSU" if (p_dep > 0.65 and dep_orani <= 1.85) else (
            "🔥 GÜVENLİ BANKO" if p_dep > 0.64 else "⭐ DEPLASMAN ÖNDE"
        )
        en_iyi_ev_skor = int(np.floor(ev_final_xg))
        en_iyi_dep_skor = int(np.ceil(dep_final_xg)) if dep_final_xg > ev_final_xg else int(np.floor(dep_final_xg)) + 1
        if en_iyi_dep_skor <= en_iyi_ev_skor:
            en_iyi_dep_skor = en_iyi_ev_skor + 1

    else:
        ana_tahmin = "Beraberlik (X)"
        secilen_olasilik = p_ber
        secilen_oran = ber_orani
        rozet = "🧩 BERABERLİK SÜRPRİZİ"
        ortalama_gol = (ev_final_xg + dep_final_xg) / 2
        en_iyi_ev_skor = int(round(ortalama_gol))
        en_iyi_dep_skor = en_iyi_ev_skor

    # --- VALUE BET BAŞARI OPTİMİZASYONU VE GÜVENLİ KASA YÖNETİMİ ---
    is_value = False
    kelly_yuzde = 0.0

    if (secilen_olasilik * secilen_oran) > 1.06:
        is_value = True
        b = secilen_oran - 1
        if b > 0:
            ham_kelly = ((b * secilen_olasilik) - (1 - secilen_olasilik)) / b
            kelly_yuzde = max(0.0, ham_kelly) * 0.5 * 100

    # --- KATEGORİSEL SÜRPRİZ VE DURUM DEDEKTÖRLERİ ---
    is_6_gol = False
    is_iy_ms_ters = False
    is_iy_ms_kilit = False
    is_duello_kg = False
    durum_notu = "Normal Seyir"

    # 1. Bol Gol ve Kaos Adayı Kontrolü
    if p_alti_arti_final > 0.18 and (ev_final_xg + dep_final_xg) >= 3.80:
        is_6_gol = True
        durum_notu = "💣 +6 GOL YAĞMURU ADAYI"
        if en_büyük == p_ev:
            en_iyi_ev_skor, en_iyi_dep_skor = 4, 2
        elif en_büyük == p_dep:
            en_iyi_ev_skor, en_iyi_dep_skor = 2, 4
        else:
            en_iyi_ev_skor, en_iyi_dep_skor = 3, 3

    # 2. İY/MS Ters Köşe Dedektörü
    if abs(p_ev - p_dep) < 0.08 and lig_kaos['kaos_faktoru'] >= 1.15 and (ev_orani >= 2.60 or dep_orani >= 2.60):
        is_iy_ms_ters = True
        durum_notu = "🔄 İY: 2 / MS: 1 (Ters Köşe)" if ev_orani < dep_orani else "🔄 İY: 1 / MS: 2 (Ters Köşe)"

    # 3. İY/MS Kilit Maç Dedektörü
    if ber_orani < 3.40 and abs(ev_final_xg - dep_final_xg) < 0.40 and p_ber > 0.30:
        is_iy_ms_kilit = True
        durum_notu = "🧩 İY: 1 / MS: 0 (Kilit)" if ev_final_xg > dep_final_xg else "🧩 İY: 2 / MS: 0 (Kilit)"

    # 4. Karşılıklı Gol Düellosu Kontrolü (İki Yarıda da KG Var)
    if p_kg_var > 0.68 and ev_final_xg > 1.60 and dep_final_xg > 1.60:
        is_duello_kg = True
        durum_notu = "⚽ DÜELLO: İY KG VAR & MS KG VAR"

        # 🛠️ Skor Çelişkisi Düzeltmesi: Her iki yarıda da KG çıkması için nihai skorlar minimum 2-2 olmalıdır.
        if en_iyi_ev_skor < 2:
            en_iyi_ev_skor = 2
        if en_iyi_dep_skor < 2:
            en_iyi_dep_skor = 2

        # Eğer bir tarafın kazanması öngörülüyorsa beraberliği bozacak şekilde skoru yukarı esnet
        if en_büyük == p_ev and en_iyi_ev_skor == en_iyi_dep_skor:
            en_iyi_ev_skor += 1
        elif en_büyük == p_dep and en_iyi_ev_skor == en_iyi_dep_skor:
            en_iyi_dep_skor += 1

    # --- ALT / ÜST VE KG KOMBİNASYON ENTEGRASYONU ---
    gol_tahmini = "2.5 Üst" if p_ust > 0.52 else "2.5 Alt"
    kg_tahmini = "KG Var" if p_kg_var > 0.52 else "KG Yok"
    ms_harfi = '1' if en_büyük == p_ev else ('2' if en_büyük == p_dep else 'X')
    kombine_tahmin = f"MS {ms_harfi} & {gol_tahmini}"

    # Karşılıklı gol tahmini var ise standart sıfır kontrolü
    if kg_tahmini == "KG Var" and not is_duello_kg:
        if en_iyi_ev_skor == 0:
            en_iyi_ev_skor = 1
        if en_iyi_dep_skor == 0:
            en_iyi_dep_skor = 1

    # İlk Yarı (İY) Skor Modellemesi
    if is_duello_kg:
        # İki yarıda da KG var olması için ilk yarı skorunun mutlaka en az 1-1 bitmesi şarttır.
        en_iyi_iy_ev, en_iyi_iy_dep = 1, 1
    else:
        en_iyi_iy_ev = max(0, int(np.floor(en_iyi_ev_skor * 0.45)))
        en_iyi_iy_dep = max(0, int(np.floor(en_iyi_dep_skor * 0.45)))

    if is_iy_ms_ters and not is_duello_kg:
        if ev_orani < dep_orani:
            en_iyi_iy_ev, en_iyi_iy_dep = 0, 1
        else:
            en_iyi_iy_ev, en_iyi_iy_dep = 1, 0
    elif kg_tahmini == "KG Var" and en_iyi_ev_skor >= 2 and en_iyi_dep_skor >= 1 and en_iyi_iy_ev == 0 and not is_duello_kg:
        en_iyi_iy_ev = 1

    iy_skor_tahmini = f"{en_iyi_iy_ev} - {en_iyi_iy_dep}"

    # Yapay Zeka Sonuç Objelerini Paketle
    return {
        "tarih": tarih_metni, "lig": lig, "mac": f"{ev} - {dep}", "rozet": rozet,
        "tahmin": ana_tahmin, "gol_tahmin": gol_tahmini, "kg_tahmin": kg_tahmini,
        "iy_skor": iy_skor_tahmini, "kombine": kombine_tahmin,
        "olasilik": secilen_olasilik * 100, "skor": f"{en_iyi_ev_skor} - {en_iyi_dep_skor}",
        "oranlar": f"1: {ev_orani:.2f} | X: {ber_orani:.2f} | 2: {dep_orani:.2f}", "ekstra": durum_notu,
        "is_value": is_value, "kelly_yuzde": kelly_yuzde, "cat_6_gol": is_6_gol,
        "cat_iy_ms_ters": is_iy_ms_ters, "cat_iy_ms_kilit": is_iy_ms_kilit, "cat_duello_kg": is_duello_kg
    }


# ==============================================================================
# 🔄 3. ADIM: GÜNCEL MAÇ DÖNGÜSÜ VE SİDEBAR LİG FİLTRESİ (BÖLÜM A)
# ==============================================================================
if not df_ham.empty:
    tum_analizler = []

    # 📖 LİG KISALTMALARINI ANLAŞILIR İSİMLERE ÇEVİREN SÖZLÜK
    lig_isimleri_sozlugu = {
        "E0": "İngiltere Premier Lig (E0)",
        "E1": "İngiltere Championship (E1)",
        "E2": "İngiltere Lig 1 (E2)",
        "E3": "İngiltere Lig 2 (E3)",
        "EC": "İngiltere Ulusal Lig (EC)",
        "D1": "Almanya Bundesliga 1 (D1)",
        "D2": "Almanya Bundesliga 2 (D2)",
        "SP1": "İspanya La Liga 1 (SP1)",
        "SP2": "İspanya La Liga 2 (SP2)",
        "I1": "İtalya Serie A (I1)",
        "I2": "İtalya Serie B (I2)",
        "F1": "Fransa Ligue 1 (F1)",
        "F2": "Fransa Ligue 2 (F2)",
        "B1": "Belçika Pro League (B1)",
        "N1": "Hollanda Eredivisie (N1)",
        "P1": "Portekiz Liga 1 (P1)",
        "T1": "Türkiye Süper Lig (T1)",
        "G1": "Yunanistan Süper Lig (G1)",
        "SCO0": "İskoçya Premier Lig (SCO0)",
        "SCO1": "İskoçya Championship (SCO1)"
    }

    # Tüm bülteni öncelikle analiz motorundan geçir
    for _, row in df_ham.iterrows():
        tarih_degeri = ""
        if 'Date' in row and not pd.isna(row['Date']):
            try:
                parsed_date = pd.to_datetime(
                    row['Date'],
                    dayfirst=True,
                    errors='coerce',
                    format='mixed'
                )
                if not pd.isna(parsed_date):
                    tarih_degeri = parsed_date.strftime('%d.%m.%Y')
                else:
                    tarih_degeri = str(row['Date'])
            except:
                tarih_degeri = str(row['Date'])

        # Maçı analiz et
        analiz = ultra_surpriz_tahmin_motoru(
            row['Div'], row['HomeTeam'], row['AwayTeam'],
            row['B365H'], row['B365D'], row['B365A'], tarih_degeri
        )
        tum_analizler.append(analiz)

    # 🗺️ SİDEBAR LİG SEÇİM PANELİ (Geliştirilmiş Anlaşılır İsimlerle)
    mevcut_kisa_ligler = sorted(list(set([m['lig'] for m in tum_analizler])))

    gosterilecek_lig_isimleri = []
    ters_sozluk = {}

    for kisa_kod in mevcut_kisa_ligler:
        uzun_isim = lig_isimleri_sozlugu.get(kisa_kod, f"Diğer Lig ({kisa_kod})")
        gosterilecek_lig_isimleri.append(uzun_isim)
        ters_sozluk[uzun_isim] = kisa_kod

    lig_secenekleri = ["🌍 Tüm Ligler"] + gosterilecek_lig_isimleri

    secilen_uzun_lig = st.sidebar.selectbox(
        "🏆 Lig Filtresi Seçin",
        options=lig_secenekleri,
        index=0,
        help="Sadece seçtiğiniz lige ait analizleri görüntüler."
    )

    # Seçilen lige göre filtreleme yap
    if secilen_uzun_lig != "🌍 Tüm Ligler":
        kisa_kod_filtresi = ters_sozluk[secilen_uzun_lig]
        aktif_analizler = [m for m in tum_analizler if m['lig'] == kisa_kod_filtresi]
        ekran_lig_adi = secilen_uzun_lig
    else:
        aktif_analizler = tum_analizler
        ekran_lig_adi = "🌍 Tüm Ligler"

    # Filtrelenmiş bültene göre listeleri doldur
    list_6_gol = []
    list_iy_ms_ters = []
    list_iy_ms_kilit = []
    list_duello_kg = []
    list_value_bets = []

    for analiz in aktif_analizler:
        if analiz["cat_6_gol"]:
            list_6_gol.append(analiz)
        if analiz["cat_iy_ms_ters"]:
            list_iy_ms_ters.append(analiz)
        if analiz["cat_iy_ms_kilit"]:
            list_iy_ms_kilit.append(analiz)
        if analiz["cat_duello_kg"]:
            list_duello_kg.append(analiz)
        if analiz["is_value"]:
            list_value_bets.append(analiz)
    # ==============================================================================
    # 🔄 3. ADIM: SEKME YAPISI VE GÖRSEL PANEL ÇIKTILARI (BÖLÜM B)
    # ==============================================================================
    # Aktif Bilgilendirme Başlığı ve Sekmeler
    st.write(f"### 📊 Filtre Durumu: `{ekran_lig_adi}` ({len(aktif_analizler)} Maç Gösteriliyor)")

    sk_tum, sk_value, sk_6, sk_ters, sk_kilit, sk_kg = st.tabs([
        "📋 Tüm Bülten", "💰 Değerli Bahisler (Value)", "💣 Gol Yağmurları (+6)",
        "🔄 İY/MS Ters Köşe", "🧩 İY/MS Kilitlenme", "⚽ İki Yarıda KG Var"
    ])

    # 📑 SEKME 1: TÜM GLOBAL BÜLTEN
    with sk_tum:
        st.subheader("📋 Seçili Ligdeki Maç Analizleri")
        for m in aktif_analizler:
            tarih_eki = f" [{m['tarih']}]" if m['tarih'] else ""
            uzun_lig_adi = lig_isimleri_sozlugu.get(m['lig'], m['lig'])
            with st.expander(f"🏟️ Lig: {uzun_lig_adi} | {m['mac']}{tarih_eki} — {m['rozet']}"):
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"🎯 **Ana Tahmin:** `{m['tahmin']}`")
                    st.markdown(f"⚽ **Gol / KG:** `{m['gol_tahmin']}` / `{m['kg_tahmin']}`")
                    st.markdown(f"📊 **Kombine Tahmin:** `{m['kombine']}`")
                with col_right:
                    st.markdown(f"⏱️ **Hedef Skorlar:** İlk Yarı: `{m['iy_skor']}` | Maç Sonucu: `{m['skor']}`")
                    st.markdown(f"💰 **Piyasa Oranları:** `{m['oranlar']}`")
                    st.markdown(f"📈 **Yapay Zeka Güven Olasılığı:** `%{m['olasilik']:.1f}`")
                if m['is_value']:
                    st.success(f"🔥 **Value Bet Saptandı!** Güvenli Kasa Yatırımı (Yarı-Kelly): **%{m['kelly_yuzde']:.2f}**")

    # 📑 SEKME 2: DEĞERLİ BAHİSLER (VALUE)
    with sk_value:
        st.subheader("💰 Bahis Şirketinin Hatalı Açtığı Değerli Oranlar")
        if list_value_bets:
            for m in list_value_bets:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                uzun_lig_adi = lig_isimleri_sozlugu.get(m['lig'], m['lig'])
                st.markdown(f"<div style='background-color: #e8f5e9; padding: 15px; border-left: 5px solid #2e7d32; border-radius: 6px; margin-bottom: 10px;'><b style='color: #2e7d32;'>💎 [{uzun_lig_adi}] VALUE MAÇ:</b> {m['mac']}{tarih_eki}</div>", unsafe_allow_html=True)
                with st.expander(f"🔍 Kasa Yönetim Detayları: {m['mac']}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Yapay Zeka Seçimi:** {m['tahmin']} (%{m['olasilik']:.1f} İhtimal)")
                        st.write(f"**Kombine Öneri:** {m['kombine']} | **Hedef Skor:** {m['skor']}")
                    with c2:
                        st.write(f"**Piyasa Oranları:** {m['oranlar']}")
                        st.info(f"📊 **Yarı-Kelly Kriteri Tavsiyesi:** Bu maça ana kasanızın en fazla **%{m['kelly_yuzde']:.2f}** miktarını ayırmalısınız.")
        else:
            st.info("Seçili filtrede oran/ihtimal dengesizliği yaratan değerli (Value) bahis bulunamadı.")

    # 📑 SEKME 3: GOL YAĞMURLARI (+6)
    with sk_6:
        st.subheader("💣 Global +6 Gol Potansiyeli")
        if list_6_gol:
            for m in list_6_gol:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                uzun_lig_adi = lig_isimleri_sozlugu.get(m['lig'], m['lig'])
                st.markdown(f"<div style='background-color: #ffebee; padding: 15px; border-left: 5px solid #c62828; border-radius: 6px; margin-bottom: 10px;'><b style='color: #c62828;'>🔥 [{uzun_lig_adi}] BOL GOL ADAYI:</b> {m['mac']}{tarih_eki}</div>", unsafe_allow_html=True)
                with st.expander(f"📋 Maç Detayları: {m['mac']}"):
                    st.write(f"**Öngörülen Skor Akışı:** İlk Yarı: `{m['iy_skor']}` ➔ Maç Sonucu: `{m['skor']}`")
                    st.write(f"**Piyasa Oranları:** {m['oranlar']}")
        else:
            st.info("Seçili filtrede +6 Gol eşiğini geçebilen ekstrem maç bulunamadı.")

    # 📑 SEKME 4: İY/MS TERS KÖŞE
    with sk_ters:
        st.subheader("🔄 Global 1/2 veya 2/1 Dönüş Sürprizleri")
        if list_iy_ms_ters:
            for m in list_iy_ms_ters:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                uzun_lig_adi = lig_isimleri_sozlugu.get(m['lig'], m['lig'])
                st.markdown(f"<div style='background-color: #fff3e0; padding: 15px; border-left: 5px solid #ef6c00; border-radius: 6px; margin-bottom: 10px;'><b style='color: #ef6c00;'>⚡ [{uzun_lig_adi}] TERS KÖŞE (1/2 - 2/1 Adayı):</b> {m['mac']}{tarih_eki}</div>", unsafe_allow_html=True)
                with st.expander(f"🔮 Risk Senaryosu: {m['mac']}"):
                    st.markdown(f"**Yapay Zeka Algoritma Notu:** `{m['ekstra']}`")
                    st.markdown(f"📋 **Tahmini Skor Gelişimi:** İlk Yarı: `{m['iy_skor']}` | Maç Sonucu: `{m['skor']}`")
                    st.write(f"**Oranlar:** {m['oranlar']}")
        else:
            st.info("Seçili filtrede 1/2 veya 2/1 riski taşıyan sürpriz maç bulunamadı.")

    # 📑 SEKME 5: İY/MS KİLİTLENME
    with sk_kilit:
        st.subheader("🧩 Global 1/0 veya 2/0 Kilitlenme Sürprizleri")
        if list_iy_ms_kilit:
            for m in list_iy_ms_kilit:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                uzun_lig_adi = lig_isimleri_sozlugu.get(m['lig'], m['lig'])
                st.markdown(f"<div style='background-color: #e1f5fe; padding: 15px; border-left: 5px solid #0277bd; border-radius: 6px; margin-bottom: 10px;'><b style='color: #0277bd;'>🔒 [{uzun_lig_adi}] KİLİTLENME:</b> {m['mac']}{tarih_eki}</div>", unsafe_allow_html=True)
                with st.expander(f"🧩 Kapanma Analiz Detayı: {m['mac']}"):
                    st.markdown(f"**Yapay Zeka Tahmini:** `{m['ekstra']}`")
                    st.write(f"**Oranlar:** {m['oranlar']}")
        else:
            st.info("Seçili filtrede ilk yarıyı önde kapatıp maçı beraberlikle bitirecek kilit bir maç saptanmadı.")

    # 📑 SEKME 6: İKİ YARIDA KG VAR
    with sk_kg:
        st.subheader("⚽ Global İki Yarıda da KG Var Düelloları")
        if list_duello_kg:
            for m in list_duello_kg:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                uzun_lig_adi = lig_isimleri_sozlugu.get(m['lig'], m['lig'])
                st.markdown(f"<div style='background-color: #f3e5f5; padding: 15px; border-left: 5px solid #8e24aa; border-radius: 6px; margin-bottom: 10px;'><b style='color: #8e24aa;'>🎯 [{uzun_lig_adi}] İKİ YARIDA GOL DÜELLOSU:</b> {m['mac']}{tarih_eki}</div>", unsafe_allow_html=True)
                with st.expander(f"⚔️ Kombinasyon Detayları: {m['mac']}"):
                    st.write(f"**Beklenen Skor:** {m['skor']} | **Oranlar:** {m['oranlar']}")
        else:
            st.info("Seçili filtrede iki yarıda da yüksek olasılıkla KG VAR çıkabilecek bir maç saptanmadı.")
else:
    st.warning("⚠️ Analiz edilecek güncel maç verisi bulunamadı. Lütfen 'guncel.csv' dosyasını kontrol edin.")
