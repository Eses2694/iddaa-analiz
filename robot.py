import streamlit as st
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# Web Tasarım Ayarları
st.set_page_config(page_title="AI İddaa Global Analyzer v6.0", page_icon="🔮", layout="wide")
st.title("🔮 Ultra Yapay Zeka Sürpriz & Evrensel Analiz Merkezi")
st.markdown("**Sürüm 6.0 Tüm Ligler Destekli:** Tüm Geçmiş Veri (11.000+ Maç) ve Ligler Otomatik Modellenir.")
st.markdown("---")

# --- 1. ADIM: VERİ OKUMA VE SÜRPRİZ ENDEKSİ HESAPLAMA ---
ihtiyacimiz_olan_sutunlar = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']

try:
    df_gecmis = pd.read_csv('maclar.csv', usecols=ihtiyacimiz_olan_sutunlar)
    df_gecmis = df_gecmis.dropna()
    df_gecmis['Date'] = pd.to_datetime(df_gecmis['Date'], dayfirst=True, errors='coerce', format='mixed')
    df_gecmis = df_gecmis.sort_values(by='Date').reset_index(drop=True)

    toplam_mac = len(df_gecmis)
    df_gecmis['Ağırlık'] = np.exp(6 * (df_gecmis.index / toplam_mac)) / np.exp(6)

except FileNotFoundError:
    st.error("❌ 'maclar.csv' dosyası bulunamadı! Lütfen dosya adının tam olarak 'maclar.csv' olduğundan emin olun.")
    st.stop()

# --- DİNAMİK LİG SÜRPRİZ VE BOL GOL ENDEKSLERİ (Excel'deki Tüm Ligler İçin) ---
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
    ratingler = {}
    for lig, ort in lig_istatistikleri.items():
        df_lig = df[df['Div'] == lig]
        takimlar = set(df_lig['HomeTeam'].unique()).union(set(df_lig['AwayTeam'].unique()))

        for takim in takimlar:
            df_ev = df_lig[df_lig['HomeTeam'] == takim]
            df_dep = df_lig[df_lig['AwayTeam'] == takim]

            if not df_ev.empty:
                w_ev = df_ev['Ağırlık'].sum()
                hucum_ic = ((df_ev['FTHG'] * df_ev['Ağırlık']).sum() / w_ev) / ort['ev_ort']
                savunma_ic = ((df_ev['FTAG'] * df_ev['Ağırlık']).sum() / w_ev) / ort['dep_ort']
            else:
                hucum_ic, savunma_ic = 1.0, 1.0

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

# Üst Bilgi Kartları
col1, col2 = st.columns(2)
col1.metric("📚 Yapay Zekanın Hafızası (Toplam Maç)", f"{len(df_gecmis)} Geçmiş Maç İncelendi")

try:
    df_ham = pd.read_csv('guncel.csv')
    for col in df_ham.columns:
        if col.lower() == 'date':
            df_ham = df_ham.rename(columns={col: 'Date'})

    df_ham = df_ham.dropna(subset=['Div', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A'])
    col2.metric("🔮 İncelenecek Güncel Maç", f"{len(df_ham)} Maç")
except FileNotFoundError:
    df_ham = pd.DataFrame()
    col2.metric("🔮 İncelenecek Güncel Maç", "0 Maç (guncel.csv eksik)")


# --- SÜPRİZ DEDEKTÖRLÜ GELİŞMİŞ TAHMİN MOTORU ---
def ultra_surpriz_tahmin_motoru(lig, ev, dep, ev_orani, ber_orani, dep_orani, tarih_metni):
    lig, ev, dep = str(lig), str(ev), str(dep)
    ev_orani, ber_orani, dep_orani = float(ev_orani), float(ber_orani), float(dep_orani)

    # Excel'den gelen yeni/farklı bir lig ise global ortalamaları otomatik oluşturarak kilitlenmeyi engelle
    ort = lig_istatistikleri.get(lig, {'ev_ort': 1.35, 'dep_ort': 1.05})
    ev_r = takim_veritabani.get((lig, ev), {'hucum_ic': 1.0, 'savunma_ic': 1.0, 'hucum_dis': 1.0, 'savunma_dis': 1.0})
    dep_r = takim_veritabani.get((lig, dep), {'hucum_ic': 1.0, 'savunma_ic': 1.0, 'hucum_dis': 1.0, 'savunma_dis': 1.0})
    lig_kaos = lig_surpriz_katsayilari.get(lig, {'alti_arti_egilimi': 0.1, 'kaos_faktoru': 1.05})

    ev_xg = ev_r['hucum_ic'] * dep_r['savunma_dis'] * ort['ev_ort']
    dep_xg = dep_r['hucum_dis'] * ev_r['savunma_ic'] * ort['dep_ort']

    toplam_oran_gucu = (1.0 / ev_orani) + (1.0 / ber_orani) + (1.0 / dep_orani)
    ev_piyasa = ((1.0 / ev_orani) / toplam_oran_gucu) * (ort['ev_ort'] + ort['dep_ort']) * 0.55
    dep_piyasa = ((1.0 / dep_orani) / toplam_oran_gucu) * (ort['ev_ort'] + ort['dep_ort']) * 0.45

    ev_final_xg = float(max((ev_xg * 0.80) + (ev_piyasa * 0.20), 0.15))
    dep_final_xg = float(max((dep_xg * 0.80) + (dep_piyasa * 0.20), 0.15))

    p_ev, p_ber, p_dep, p_ust, p_kg_var, p_alti_arti = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def dixon_coles_ayari(x, y, mu, lam, rho=-0.05):
        if x == 0 and y == 0: return 1 - (mu * lam * rho)
        if x == 1 and y == 0: return 1 + (mu * rho)
        if x == 0 and y == 1: return 1 + (lam * rho)
        if x == 1 and y == 1: return 1 - rho
        return 1.0

    for x in range(10):
        for y in range(10):
            skor_p = float(poisson.pmf(x, ev_final_xg) * poisson.pmf(y, dep_final_xg))
            skor_p *= dixon_coles_ayari(x, y, ev_final_xg, dep_final_xg)

            if x > y:
                p_ev += skor_p
            elif x == y:
                p_ber += skor_p
            else:
                p_dep += skor_p
            if (x + y) > 2: p_ust += skor_p
            if x >= 1 and y >= 1: p_kg_var += skor_p
            if (x + y) >= 6: p_alti_arti += skor_p

    p_alti_arti_final = p_alti_arti * lig_kaos['kaos_faktoru']

    olasılıklar = [p_ev, p_ber, p_dep]
    en_büyük = max(olasılıklar)

    if en_büyük == p_ev:
        ana_tahmin = "Ev Sahibi Kazanır (1)"
        rozet = "🔥 GÜVENLİ BANKO" if p_ev > 0.64 else "⭐ EV SAHİBİ ÖNDE"
        en_iyi_ev_skor = int(np.ceil(ev_final_xg)) if ev_final_xg > dep_final_xg else int(np.floor(ev_final_xg)) + 1
        en_iyi_dep_skor = int(np.floor(dep_final_xg))
        if en_iyi_ev_skor <= en_iyi_dep_skor: en_iyi_ev_skor = en_iyi_dep_skor + 1
    elif en_büyük == p_dep:
        ana_tahmin = "Deplasman Kazanır (2)"
        rozet = "🔥 GÜVENLİ BANKO" if p_dep > 0.64 else "⭐ DEPLASMAN ÖNDE"
        en_iyi_ev_skor = int(np.floor(ev_final_xg))
        en_iyi_dep_skor = int(np.ceil(dep_final_xg)) if dep_final_xg > ev_final_xg else int(np.floor(dep_final_xg)) + 1
        if en_iyi_dep_skor <= en_iyi_ev_skor: en_iyi_dep_skor = en_iyi_ev_skor + 1
    else:
        ana_tahmin = "Beraberlik (X)"
        rozet = "🧩 BERABERLİK SÜRPRİZİ"
        ortalama_gol = (ev_final_xg + dep_final_xg) / 2
        en_iyi_ev_skor = int(round(ortalama_gol))
        en_iyi_dep_skor = en_iyi_ev_skor

    is_6_gol = False
    is_iy_ms_ters = False
    is_iy_ms_kilit = False
    is_duello_kg = False
    durum_notu = "Normal Seyir"

    if p_alti_arti_final > 0.18 and (ev_final_xg + dep_final_xg) >= 3.80:
        is_6_gol = True
        durum_notu = "💣 +6 GOL YAĞMURU ADAYI"
        if en_büyük == p_ev:
            en_iyi_ev_skor, en_iyi_dep_skor = 4, 2
        elif en_büyük == p_dep:
            en_iyi_ev_skor, en_iyi_dep_skor = 2, 4
        else:
            en_iyi_ev_skor, en_iyi_dep_skor = 3, 3

    if abs(p_ev - p_dep) < 0.08 and lig_kaos['kaos_faktoru'] >= 1.15 and (ev_orani >= 2.60 or dep_orani >= 2.60):
        is_iy_ms_ters = True
        durum_notu = "🔄 İY: 2 / MS: 1 (Ters Köşe)" if ev_orani < dep_orani else "🔄 İY: 1 / MS: 2 (Ters Köşe)"

    if ber_orani < 3.40 and abs(ev_final_xg - dep_final_xg) < 0.40 and p_ber > 0.30:
        is_iy_ms_kilit = True
        durum_notu = "🧩 İY: 1 / MS: 0 (Kilit)" if ev_final_xg > dep_final_xg else "🧩 İY: 2 / MS: 0 (Kilit)"

    if p_kg_var > 0.68 and ev_final_xg > 1.60 and dep_final_xg > 1.60:
        is_duello_kg = True
        durum_notu = "⚽ DÜELLO: İY KG VAR & MS KG VAR"

    return {
        "tarih": tarih_metni,
        "lig": lig,  # Dinamik lig kodu doğrudan basılıyor
        "mac": f"{ev} - {dep}",
        "rozet": rozet,
        "tahmin": ana_tahmin,
        "skor": f"{en_iyi_ev_skor} - {en_iyi_dep_skor}",
        "oranlar": f"1: {ev_orani} | X: {ber_orani} | 2: {dep_orani}",
        "ekstra": durum_notu,
        "cat_6_gol": is_6_gol,
        "cat_iy_ms_ters": is_iy_ms_ters,
        "cat_iy_ms_kilit": is_iy_ms_kilit,
        "cat_duello_kg": is_duello_kg
    }


# --- --- 2. ADIM: EKRANDA SEKME YAPISI OLUŞTURMA VE GÖSTERİM --- ---
if not df_ham.empty:
    tum_analizler = []
    list_6_gol = []
    list_iy_ms_ters = []
    list_iy_ms_kilit = []
    list_duello_kg = []

    for _, row in df_ham.iterrows():
        tarih_degeri = ""
        if 'Date' in row and not pd.isna(row['Date']):
            try:
                parsed_date = pd.to_datetime(row['Date'], dayfirst=True, errors='coerce', format='mixed')
                if not pd.isna(parsed_date):
                    tarih_degeri = parsed_date.strftime('%d.%m.%Y')
                else:
                    tarih_degeri = str(row['Date'])
            except:
                tarih_degeri = str(row['Date'])

        analiz = ultra_surpriz_tahmin_motoru(
            row['Div'], row['HomeTeam'], row['AwayTeam'],
            row['B365H'], row['B365D'], row['B365A'],
            tarih_degeri
        )
        tum_analizler.append(analiz)

        if analiz["cat_6_gol"]: list_6_gol.append(analiz)
        if analiz["cat_iy_ms_ters"]: list_iy_ms_ters.append(analiz)
        if analiz["cat_iy_ms_kilit"]: list_iy_ms_kilit.append(analiz)
        if analiz["cat_duello_kg"]: list_duello_kg.append(analiz)

    # 5 Temiz Sekme Yapısı
    sk_tum, sk_6, sk_ters, sk_kilit, sk_kg = st.tabs([
        "📋 Tüm Global Bülten",
        "💣 Gol Yağmurları (+6)",
        "🔄 İY/MS Ters Köşe (1/2, 2/1)",
        "🧩 İY/MS Kilitlenme (1/0, 2/0)",
        "⚽ İki Yarıda da KG Var"
    ])

    # 1. TÜM GLOBAL BÜLTEN
    with sk_tum:
        st.subheader("📋 Excel Dosyasındaki Tüm Maç Analizleri")
        for m in tum_analizler:
            tarih_eki = f" [{m['tarih']}]" if m['tarih'] else ""
            with st.expander(f"Lig: {m['lig']} | {m['mac']}{tarih_eki}"):
                st.write(f"**Tahmin:** {m['tahmin']} | **Hedef Skor:** {m['skor']}")
                st.write(f"**Oranlar:** {m['oranlar']}")

    # 2. GOL YAĞMURLARI
    with sk_6:
        st.subheader("💣 Global +6 Gol Potansiyeli")
        if list_6_gol:
            for m in list_6_gol:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                st.error(f"🔥 [{m['lig']}] BOL GOL: {m['mac']}{tarih_eki}")
                with st.expander(f"Detaylar: {m['mac']}"):
                    st.write(f"**Hedef Skor:** {m['skor']} | **Oranlar:** {m['oranlar']}")
        else:
            st.info("İnce eleme başarılı: Tüm bültende +6 Gol eşiğini geçebilen ekstrem maç bulunamadı.")

    # 3. İY/MS TERS KÖŞE (1/2 - 2/1)
    with sk_ters:
        st.subheader("🔄 Global 1/2 veya 2/1 Dönüş Sürprizleri")
        if list_iy_ms_ters:
            for m in list_iy_ms_ters:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                st.warning(f"⚡ [{m['lig']}] TERS KÖŞE: {m['mac']}{tarih_eki}")
                with st.expander(f"Senaryo: {m['mac']}"):
                    st.markdown(f"**Yapar Zeka Kodu:** `{m['ekstra']}`")
                    st.write(f"**Oranlar:** {m['oranlar']}")
        else:
            st.info("İnce eleme başarılı: Bültende 1/2 veya 2/1 riski taşıyan elit maç bulunamadı.")

    # 4. İY/MS KİLİTLENME (1/0 - 2/0)
    with sk_kilit:
        st.subheader("🧩 Global 1/0 veya 2/0 Kilitlenme Sürprizleri")
        if list_iy_ms_kilit:
            for m in list_iy_ms_kilit:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                st.info(f"🔒 [{m['lig']}] KİLİTLENME: {m['mac']}{tarih_eki}")
                with st.expander(f"Kapanma Detayı: {m['mac']}"):
                    st.markdown(f"**Yapay Zeka Tahmini:** `{m['ekstra']}`")
                    st.write(f"**Oranlar:** {m['oranlar']}")
        else:
            st.info("İnce eleme başarılı: İlk yarıyı önde kapatıp maçı beraberlikle bitirecek kilit bir maç yok.")

    # 5. İKİ YARIDA DA KG VAR
    with sk_kg:
        st.subheader("⚽ Global İki Yarıda da KG Var Düelloları")
        if list_duello_kg:
            for m in list_duello_kg:
                tarih_eki = f" ({m['tarih']})" if m['tarih'] else ""
                st.success(f"🎯 [{m['lig']}] İKİ YARIDA GOL: {m['mac']}{tarih_eki}")
                with st.expander(f"Kombinasyon: {m['mac']}"):
                    st.write(f"**Beklenen Skor:** {m['skor']} | **Oranlar:** {m['oranlar']}")
        else:
            st.info("İnce eleme başarılı: İki yarıda da yüksek olasılıkla KG VAR çıkabilecek bir maç saptanmadı.")
else:
    st.warning("Analiz edilecek güncel maç verisi yok. 'guncel.csv' dosyasını kontrol edin.")
