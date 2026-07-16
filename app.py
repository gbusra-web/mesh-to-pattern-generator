# -*- coding: utf-8 -*-
"""
arayuz.py
---------
Streamlit tabanlı arayüz: STL -> terzi kalıbı iş akışı.

Çalıştırma:
    streamlit run app.py
"""

import io
import zipfile

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

from kalip import (
    mesh_yukle,
    mesh_olceklendir,  # <--- HATA VEREN ENTEGRASYON BURAYA EKLENDİ
    komsuluk_matrisi_olustur,
    mesh_segmentlere_ayir,
    segment_renkleri_uret,
    parcalari_uret,
    toplam_distorsiyon_skoru,
    optimal_parca_sayisi_bul,
)

st.set_page_config(page_title="Terzi Kalıbı Çıkarıcı", layout="wide")

BIRIM_CARPANLARI = {
    "mm (varsayılan)": 1.0,
    "cm": 0.01,   # alan mm^2 -> cm^2 : /100
    "inch": 1.0 / 645.16,  # mm^2 -> inch^2
}

UZUNLUK_CARPANLARI = {
    "mm (varsayılan)": 1.0,
    "cm": 0.1,   # mm'yi cm'ye çevirmek için 10'a böler
    "inch": 1.0 / 25.4, 
}

# ---------------------------------------------------------------------------
# Yardımcı: mesh'i önbelleğe al
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def _mesh_yukle_cache(dosya_bytes, dosya_adi):
    import tempfile, os
    suffix = os.path.splitext(dosya_adi)[1] or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(dosya_bytes)
        gecici_yol = f.name
    mesh = mesh_yukle(gecici_yol)
    return mesh


def alan_birimine_cevir(deger_mm2, birim_adi):
    return deger_mm2 * BIRIM_CARPANLARI[birim_adi]


def uc_boyutlu_gorseli_ciz(mesh, kume_etiketleri, aktif_kumeler):
    """Plotly ile küme renklerine göre boyalı, döndürülebilir 3D model."""
    _, renk_tablosu = segment_renkleri_uret(kume_etiketleri, aktif_kumeler)
    yuz_renkleri = []
    renk_map = {k_id: renk_tablosu[i % len(renk_tablosu)] for i, k_id in enumerate(aktif_kumeler)}
    for etiket in kume_etiketleri:
        r, g, b = renk_map[etiket]
        yuz_renkleri.append(f"rgb({r},{g},{b})")

    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                facecolor=yuz_renkleri,
                flatshading=True,
                showscale=False,
            )
        ]
    )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
    )
    return fig


def iki_boyutlu_kaliplari_ciz(parcalar, birim_adi):
    """Her parça için ayrı ayrı matplotlib figürü üretir (PNG bytes olarak)."""
    import io
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    goruntuler = {}
    for i, p in enumerate(parcalar, start=1):
        poligon = p["poligon"]
        
        # 1. Hata almamak için güvenli uzunluk çarpanı tanımlaması
        uzunluk_carpanlari = {
            "mm (varsayılan)": 1.0,
            "cm": 0.1,
            "inch": 1.0 / 25.4
        }
        uzunluk_carpan = uzunluk_carpanlari.get(birim_adi, 1.0)
        birim_kisa = birim_adi.split(" ")[0]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        ax.set_axisbelow(True)

        # 2. Gerçek Kalıp (Dikiş Çizgisi - Siyah)
        x, y = poligon.exterior.xy
        ax.plot(x, y, color="black", linewidth=2.5, label="Dikiş Çizgisi (Asıl Boyut)")
        ax.fill(x, y, color="#f0f0f0", alpha=0.5)

        # --- YENİ: TERZİ İÇİN REFERANS NOKTALARI (ÇIT İŞARETLERİ) ---
        toplam_nokta = len(x)
        adim = max(1, toplam_nokta // 12)
        for idx in range(0, toplam_nokta - 1, adim):
            ax.text(x[idx], y[idx], str(idx), color="purple", fontsize=9, fontweight="bold",
                    ha="center", va="center", bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

        # 3. Dikiş Payı Ekleme (Kesim Çizgisi - Mavi Kesik Çizgi)
        if "mm" in birim_adi: pay = 10.0
        elif "cm" in birim_adi: pay = 1.0
        else: pay = 0.4
            
        try:
            dikis_payli_poligon = poligon.buffer(pay, join_style=2)
            xd, yd = dikis_payli_poligon.exterior.xy
            ax.plot(xd, yd, color="blue", linestyle="-.", linewidth=1.5, label=f"Kesim Çizgisi (+{pay} {birim_kisa} Pay)")
            minx, miny, maxx, maxy = dikis_payli_poligon.bounds
        except:
            minx, miny, maxx, maxy = poligon.bounds
        
        # 4. Kumaş Kesim Ölçüleri (Bounding Box - Kırmızı)
        genislik = (maxx - minx) * uzunluk_carpan
        yukseklik = (maxy - miny) * uzunluk_carpan

        rect = patches.Rectangle((minx, miny), maxx-minx, maxy-miny, 
                                 linewidth=1.5, edgecolor='red', facecolor='none', linestyle='dashed')
        ax.add_patch(rect)
        
        # En ve Boy yazılarını ekle
        ax.text(minx + (maxx-minx)/2, miny - (maxy-miny)*0.03, f"En: {genislik:.1f} {birim_kisa}", 
                color='red', ha='center', va='top', fontsize=11, fontweight='bold')
        ax.text(minx - (maxx-minx)*0.03, miny + (maxy-miny)/2, f"Boy: {yukseklik:.1f} {birim_kisa}", 
                color='red', ha='right', va='center', fontsize=11, fontweight='bold', rotation=90)

        # 5. Başlık ve İsimlendirme
        alan_carpani = 1.0 if "mm" in birim_adi else (0.01 if "cm" in birim_adi else 1.0/645.16)
        kalip_alan = p.get("kalip_2d_alan", 0) * alan_carpani

        ax.set_title(
            f"Parça {i}\n"
            f"Gerekli Kumaş Kesimi (En x Boy): {genislik:.1f} x {yukseklik:.1f} {birim_kisa}\n"
            f"Kalıp Alanı: {kalip_alan:.1f} {birim_kisa}² | Distorsiyon: %{p.get('distorsiyon', 0)*100:.1f}",
            fontsize=11, fontweight="bold",
        )
        
        ax.axis("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        
        ax.legend(loc="upper right", fontsize=8)

        # Görüntüyü kaydet ve belleğe al
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        goruntuler[f"kalip_parca_{i}.png"] = buf.getvalue()
        
    return goruntuler

# ---------------------------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------------------------

st.title("🐝 3D → Terzi Kalıbı Çıkarıcı")

with st.sidebar:
    st.header("1. Model")
    yuklenen_dosya = st.file_uploader("STL / OBJ / PLY yükle", type=["stl", "obj", "ply"])
    birim = st.selectbox("Alan birimi", list(BIRIM_CARPANLARI.keys()))
    st.caption("dosyanız mm cinsindense 'mm' seçin; farklıysa çıktıyı buna göre okuyun.")
    
    # --- YENİ: Hata Veren Eski Çarpan Menüsü Tamamen Değiştirildi ---
    st.header("2. Fiziksel Boyutlandırma")
    hedef_cap_cm = st.slider(
        "Taban Çapı (cm)", 
        min_value=15.0, 
        max_value=70.0, 
        value=22.0, 
        step=0.5,
        help="Yetişkin bir insan kafası için ortalama 22 cm idealdir. Model orijinal halinde yüklenip bu çapa göre ölçeklendirilecektir."
    )


if yuklenen_dosya is None:
    st.info("Devam etmek için sol menüden bir STL / OBJ / PLY dosyası yükleyin.")
    st.stop()

# Orijinal modeli önbellekten yükle
orijinal_mesh = _mesh_yukle_cache(yuklenen_dosya.getvalue(), yuklenen_dosya.name)

# --- YENİ: Modelin dinamik ölçeklendiği ve hatanın çözüldüğü ana kısım ---
mesh = orijinal_mesh.copy()
mesh, uygulanan_olcek, orjinal_cap = mesh_olceklendir(mesh, hedef_cap_cm)

st.success(f"Model yüklendi. Taban ~{orjinal_cap:.2f} birimden {hedef_cap_cm} cm'ye uyarlandı. ({uygulanan_olcek:.2f} kat ölçeklendi.)")

sekme1, sekme2 = st.tabs(["1) Optimal Parça Sayısını Bul", "2) 3D Önizleme ve 2D Kalıplar"])

# --- SEKME 1: Optimal K analizi -------------------------------------------
with sekme1:
    st.subheader("Kaç parçaya bölmek en mantıklı?")
    st.write(
        "Her parça sayısı (k) için model bölünüp düzleştirilir; düzleştirilmiş "
        "(2D) alan ile gerçek kavisli yüzey alanı karşılaştırılarak bir "
        "**distorsiyon skoru** hesaplanır."
    )
    k_min, k_max = st.slider("Denenecek k aralığı", 2, 15, (2, 10))

    if st.button("Analizi Çalıştır", type="primary"):
        ilerleme = st.progress(0.0, text="Başlıyor...")

        def _ilerleme_cb(k, i, toplam):
            ilerleme.progress(i / toplam, text=f"k={k} işleniyor... ({i}/{toplam})")

        sonuclar, onerilen_k = optimal_parca_sayisi_bul(
            mesh, k_araligi=range(k_min, k_max + 1), ilerleme_callback=_ilerleme_cb
        )
        ilerleme.empty()

        st.session_state["optimal_sonuclar"] = sonuclar
        st.session_state["onerilen_k"] = onerilen_k

    if "optimal_sonuclar" in st.session_state:
        sonuclar = st.session_state["optimal_sonuclar"]
        onerilen_k = st.session_state["onerilen_k"]

        df = pd.DataFrame(sonuclar)
        df["distorsiyon_%"] = df["distorsiyon"] * 100

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["k"], y=df["distorsiyon_%"],
                    mode="lines+markers", name="Distorsiyon %",
                )
            )
            if onerilen_k is not None:
                fig.add_vline(x=onerilen_k, line_dash="dash", line_color="green")
            fig.update_layout(
                xaxis_title="Parça sayısı (k)",
                yaxis_title="Ağırlıklı distorsiyon (%)",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Önerilen parça sayısı", onerilen_k)
            st.dataframe(df[["k", "parca_sayisi_gercek", "distorsiyon_%"]], hide_index=True)

        st.info(
            f"Öneri: **k = {onerilen_k}** civarında başlayın; sonucu beğenmezseniz "
            "2. sekmede kaydırıcıdan k'yi elle ayarlayıp anlık önizleyebilirsiniz."
        )

# --- SEKME 2: 3D önizleme + 2D kalıplar -----------------------------------
with sekme2:
    varsayilan_k = st.session_state.get("onerilen_k", 6)
    k = st.slider("Parça sayısı (k)", 2, 15, int(varsayilan_k))

    with st.spinner("Segmentasyon ve düzleştirme hesaplanıyor..."):
        connectivity = komsuluk_matrisi_olustur(mesh)
        etiketler, aktif = mesh_segmentlere_ayir(mesh, k, connectivity=connectivity)
        parcalar = parcalari_uret(mesh, etiketler, aktif)
        genel_skor = toplam_distorsiyon_skoru(parcalar)

    col_3d, col_ozet = st.columns([2, 1])
    with col_3d:
        st.plotly_chart(uc_boyutlu_gorseli_ciz(mesh, etiketler, aktif), use_container_width=True)
    with col_ozet:
        st.metric("Kalıba dönüşen parça sayısı", len(parcalar))
        st.metric("Genel ağırlıklı distorsiyon", f"%{genel_skor*100:.1f}")
        if genel_skor > 0.08:
            st.warning("Distorsiyon yüksek: parça sayısını artırmayı deneyin.")
        else:
            st.success("Distorsiyon makul seviyede.")

    st.divider()
    st.subheader("2D Terzi Kalıpları")

    goruntuler = iki_boyutlu_kaliplari_ciz(parcalar, birim)
    kolonlar = st.columns(3)
    for i, (ad, veri) in enumerate(goruntuler.items()):
        with kolonlar[i % 3]:
            st.image(veri, caption=ad, use_container_width=True)

    # --- indirme: zip + csv ---
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for ad, veri in goruntuler.items():
            zf.writestr(ad, veri)
        alan_df = pd.DataFrame(
            [
                {
                    "parca": i + 1,
                    f"kalip_alan_{birim}": alan_birimine_cevir(p["kalip_2d_alan"], birim),
                    f"gercek_alan_{birim}": alan_birimine_cevir(p["gercek_3d_alan"], birim),
                    "distorsiyon_%": p["distorsiyon"] * 100,
                }
                for i, p in enumerate(parcalar)
            ]
        )
        zf.writestr("alan_tablosu.csv", alan_df.to_csv(index=False))
    zip_buf.seek(0)

    st.download_button(
        "📦 Tüm kalıpları + alan tablosunu indir (.zip)",
        data=zip_buf,
        file_name=f"terzi_kaliplari_k{k}.zip",
        mime="application/zip",
    )
