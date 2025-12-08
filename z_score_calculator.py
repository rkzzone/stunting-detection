"""
Z-Score Calculator untuk Deteksi Stunting
Berdasarkan Standar WHO Child Growth Standards
"""

import numpy as np
from scipy import interpolate

class WHOZScoreCalculator:
    """
    Kalkulator Z-Score berdasarkan standar WHO
    Menggunakan data referensi WHO untuk Height-for-Age (Tinggi Badan menurut Umur)
    """
    
    def __init__(self):
        # Data WHO Height-for-Age Standards (0-228 bulan / 0-19 tahun)
        # Format: [umur_bulan, L, M, S] untuk laki-laki dan perempuan
        
        # DATA LAKI-LAKI (Height-for-Age)
        self.male_data = {
            # 0-60 bulan (0-5 tahun)
            0: {'L': 1, 'M': 49.9, 'S': 0.03795},
            1: {'L': 1, 'M': 54.7, 'S': 0.03557},
            2: {'L': 1, 'M': 58.4, 'S': 0.03424},
            3: {'L': 1, 'M': 61.4, 'S': 0.03328},
            6: {'L': 1, 'M': 67.6, 'S': 0.03191},
            9: {'L': 1, 'M': 72.0, 'S': 0.03145},
            12: {'L': 1, 'M': 75.7, 'S': 0.03117},
            15: {'L': 1, 'M': 79.1, 'S': 0.03099},
            18: {'L': 1, 'M': 82.3, 'S': 0.03085},
            21: {'L': 1, 'M': 85.1, 'S': 0.03073},
            24: {'L': 1, 'M': 87.8, 'S': 0.03063},
            30: {'L': 1, 'M': 92.3, 'S': 0.03048},
            36: {'L': 1, 'M': 96.1, 'S': 0.03043},
            42: {'L': 1, 'M': 99.9, 'S': 0.03046},
            48: {'L': 1, 'M': 103.3, 'S': 0.03057},
            54: {'L': 1, 'M': 106.7, 'S': 0.03075},
            60: {'L': 1, 'M': 110.0, 'S': 0.03099},
            # 61-228 bulan (5-19 tahun)
            72: {'L': 1, 'M': 116.1, 'S': 0.03989},
            84: {'L': 1, 'M': 122.0, 'S': 0.04306},
            96: {'L': 1, 'M': 127.5, 'S': 0.04550},
            108: {'L': 1, 'M': 132.6, 'S': 0.04744},
            120: {'L': 1, 'M': 137.6, 'S': 0.04896},
            132: {'L': 1, 'M': 143.4, 'S': 0.05002},
            144: {'L': 1, 'M': 149.8, 'S': 0.05051},
            156: {'L': 1, 'M': 156.2, 'S': 0.05034},
            168: {'L': 1, 'M': 161.9, 'S': 0.04955},
            180: {'L': 1, 'M': 166.3, 'S': 0.04836},
            192: {'L': 1, 'M': 169.4, 'S': 0.04706},
            204: {'L': 1, 'M': 171.5, 'S': 0.04586},
            216: {'L': 1, 'M': 172.9, 'S': 0.04489},
            228: {'L': 1, 'M': 173.8, 'S': 0.04418}
        }
        
        # DATA PEREMPUAN (Height-for-Age)
        self.female_data = {
            # 0-60 bulan (0-5 tahun)
            0: {'L': 1, 'M': 49.1, 'S': 0.03790},
            1: {'L': 1, 'M': 53.7, 'S': 0.03568},
            2: {'L': 1, 'M': 57.1, 'S': 0.03445},
            3: {'L': 1, 'M': 59.8, 'S': 0.03356},
            6: {'L': 1, 'M': 65.7, 'S': 0.03244},
            9: {'L': 1, 'M': 70.1, 'S': 0.03204},
            12: {'L': 1, 'M': 74.0, 'S': 0.03175},
            15: {'L': 1, 'M': 77.5, 'S': 0.03152},
            18: {'L': 1, 'M': 80.7, 'S': 0.03133},
            21: {'L': 1, 'M': 83.7, 'S': 0.03117},
            24: {'L': 1, 'M': 86.4, 'S': 0.03104},
            30: {'L': 1, 'M': 91.3, 'S': 0.03088},
            36: {'L': 1, 'M': 95.6, 'S': 0.03085},
            42: {'L': 1, 'M': 99.8, 'S': 0.03090},
            48: {'L': 1, 'M': 103.3, 'S': 0.03103},
            54: {'L': 1, 'M': 106.7, 'S': 0.03123},
            60: {'L': 1, 'M': 109.9, 'S': 0.03149},
            # 61-228 bulan (5-19 tahun)
            72: {'L': 1, 'M': 115.5, 'S': 0.03951},
            84: {'L': 1, 'M': 121.7, 'S': 0.04280},
            96: {'L': 1, 'M': 128.2, 'S': 0.04516},
            108: {'L': 1, 'M': 135.0, 'S': 0.04643},
            120: {'L': 1, 'M': 142.0, 'S': 0.04668},
            132: {'L': 1, 'M': 148.7, 'S': 0.04615},
            144: {'L': 1, 'M': 154.2, 'S': 0.04509},
            156: {'L': 1, 'M': 157.9, 'S': 0.04378},
            168: {'L': 1, 'M': 160.1, 'S': 0.04253},
            180: {'L': 1, 'M': 161.3, 'S': 0.04152},
            192: {'L': 1, 'M': 162.0, 'S': 0.04080},
            204: {'L': 1, 'M': 162.4, 'S': 0.04033},
            216: {'L': 1, 'M': 162.6, 'S': 0.04006},
            228: {'L': 1, 'M': 162.8, 'S': 0.03994}
        }
    
    def interpolate_lms(self, age_months, gender):
        """
        Interpolasi nilai L, M, S untuk umur yang tidak ada di tabel
        Returns: (L, M, S, is_adult)
        """
        data = self.male_data if gender.lower() == 'laki-laki' else self.female_data
        
        ages = sorted(data.keys())
        is_adult = False
        
        # Jika umur di luar range, gunakan nilai terdekat
        if age_months < min(ages):
            age_months = min(ages)
        elif age_months > max(ages):
            age_months = max(ages)
            is_adult = True  # Flag untuk usia dewasa (>19 tahun)
        
        # Jika umur tepat ada di data
        if age_months in data:
            return data[age_months]['L'], data[age_months]['M'], data[age_months]['S'], is_adult
        
        # Interpolasi linear
        L_values = [data[age]['L'] for age in ages]
        M_values = [data[age]['M'] for age in ages]
        S_values = [data[age]['S'] for age in ages]
        
        L = np.interp(age_months, ages, L_values)
        M = np.interp(age_months, ages, M_values)
        S = np.interp(age_months, ages, S_values)
        
        return L, M, S, is_adult
    
    def calculate_zscore(self, age_months, height_cm, gender):
        """
        Menghitung Z-Score Height-for-Age menggunakan metode LMS WHO
        
        Formula: Z = ((height/M)^L - 1) / (L * S)
        
        Parameters:
        - age_months: Umur dalam bulan (0-228 / 0-19 tahun)
        - height_cm: Tinggi badan dalam cm
        - gender: 'Laki-laki' atau 'Perempuan'
        
        Returns:
        - zscore: Nilai Z-Score
        - is_adult: Boolean flag apakah usia > 19 tahun
        """
        L, M, S, is_adult = self.interpolate_lms(age_months, gender)
        
        # Formula WHO LMS method
        if L != 0:
            zscore = (((height_cm / M) ** L) - 1) / (L * S)
        else:
            zscore = np.log(height_cm / M) / S
        
        return round(zscore, 2), is_adult
    
    def classify_nutrition_status(self, zscore, is_adult=False):
        """
        Klasifikasi status gizi berdasarkan Z-Score
        
        Kategori WHO:
        - Severely Stunted: Z-score < -3 SD
        - Stunted: -3 SD ≤ Z-score < -2 SD
        - Normal: -2 SD ≤ Z-score ≤ +3 SD
        - Tall: Z-score > +3 SD
        
        Untuk dewasa (>19 tahun), menggunakan referensi tinggi rata-rata
        """
        if is_adult:
            # Untuk dewasa, gunakan klasifikasi berdasarkan perbandingan dengan standar akhir (19 tahun)
            if zscore < -3:
                return "Perawakan Sangat Pendek (Kemungkinan Stunting di Masa Kecil)"
            elif -3 <= zscore < -2:
                return "Perawakan Pendek (Kemungkinan Stunting di Masa Kecil)"
            elif -2 <= zscore <= 3:
                return "Perawakan Normal"
            else:
                return "Perawakan Tinggi"
        else:
            if zscore < -3:
                return "Severely Stunted (Sangat Pendek)"
            elif -3 <= zscore < -2:
                return "Stunted (Pendek)"
            elif -2 <= zscore <= 3:
                return "Normal"
            else:
                return "Tall (Tinggi)"
    
    def get_recommendation(self, zscore, status):
        """
        Memberikan rekomendasi berdasarkan status gizi
        """
        recommendations = {
            "Severely Stunted (Sangat Pendek)": {
                "title": "⚠️ PERLU PERHATIAN SERIUS",
                "actions": [
                    "Segera konsultasi ke Dokter Spesialis Anak atau Puskesmas terdekat",
                    "Anak memerlukan intervensi gizi intensif dengan Pemberian Makanan Tambahan (PMT) tinggi protein",
                    "Berikan makanan bergizi tinggi: telur, ikan, daging, tahu/tempe setiap hari",
                    "Pastikan ASI terus diberikan jika anak masih menyusui",
                    "Periksa dan obati penyakit penyerta (cacingan, TBC, diare kronis)",
                    "Lakukan stimulasi tumbuh kembang secara rutin",
                    "Pantau pertumbuhan setiap bulan di Posyandu"
                ],
                "color": "#DC2626"
            },
            "Stunted (Pendek)": {
                "title": "⚠️ PERLU PERBAIKAN GIZI",
                "actions": [
                    "Konsultasikan ke Bidan Desa atau Puskesmas untuk evaluasi lebih lanjut",
                    "Tingkatkan asupan protein hewani: minimal 1 butir telur/hari",
                    "Berikan makanan beragam dengan gizi seimbang (Isi Piringku)",
                    "Teruskan pemberian ASI hingga 2 tahun jika masih menyusui",
                    "Berikan MPASI yang tepat sesuai usia anak",
                    "Rutin ke Posyandu untuk pemantauan pertumbuhan",
                    "Jaga kebersihan dan sanitasi lingkungan"
                ],
                "color": "#F59E0B"
            },
            "Normal": {
                "title": "✅ STATUS GIZI BAIK",
                "actions": [
                    "Pertahankan pola makan bergizi seimbang",
                    "Teruskan pemberian ASI eksklusif (0-6 bulan) dan dilanjutkan hingga 2 tahun",
                    "Berikan MPASI yang bervariasi dan bergizi",
                    "Rutin ke Posyandu untuk memantau tumbuh kembang",
                    "Lengkapi imunisasi dasar sesuai jadwal",
                    "Jaga kebersihan dan pola hidup sehat",
                    "Lakukan stimulasi tumbuh kembang secara rutin"
                ],
                "color": "#10B981"
            },
            "Tall (Tinggi)": {
                "title": "✅ PERTUMBUHAN BAIK",
                "actions": [
                    "Pertumbuhan anak sangat baik, pertahankan pola asuh yang sehat",
                    "Tetap berikan makanan bergizi seimbang",
                    "Pantau tumbuh kembang secara berkala di Posyandu",
                    "Jaga aktivitas fisik anak tetap seimbang",
                    "Lengkapi imunisasi sesuai jadwal",
                    "Konsultasi rutin untuk memastikan pertumbuhan proporsional"
                ],
                "color": "#10B981"
            },
            # Rekomendasi untuk dewasa (>19 tahun)
            "Perawakan Sangat Pendek (Kemungkinan Stunting di Masa Kecil)": {
                "title": "⚠️ INDIKASI STUNTING DI MASA KECIL",
                "actions": [
                    "Tinggi badan Anda menunjukkan kemungkinan mengalami stunting di masa kecil",
                    "Konsultasi dengan dokter untuk evaluasi kesehatan secara menyeluruh",
                    "Periksa kemungkinan defisiensi nutrisi yang masih ada",
                    "Jaga pola makan bergizi seimbang untuk kesehatan optimal",
                    "Lakukan pemeriksaan kesehatan rutin",
                    "Penting untuk memastikan anak-anak Anda mendapat nutrisi yang cukup sejak dini",
                    "Konsultasi dengan ahli gizi jika merencanakan kehamilan"
                ],
                "color": "#DC2626"
            },
            "Perawakan Pendek (Kemungkinan Stunting di Masa Kecil)": {
                "title": "⚠️ KEMUNGKINAN STUNTING DI MASA KECIL",
                "actions": [
                    "Tinggi badan Anda sedikit di bawah rata-rata, kemungkinan terkait nutrisi masa kecil",
                    "Konsultasi dengan dokter untuk evaluasi kesehatan",
                    "Jaga pola makan bergizi seimbang",
                    "Lakukan pemeriksaan kesehatan rutin",
                    "Pastikan anak-anak Anda mendapat nutrisi optimal sejak dini",
                    "Konsultasi dengan ahli gizi untuk pola makan yang lebih baik"
                ],
                "color": "#F59E0B"
            },
            "Perawakan Normal": {
                "title": "✅ PERAWAKAN NORMAL",
                "actions": [
                    "Tinggi badan Anda berada dalam rentang normal",
                    "Pertahankan pola makan bergizi seimbang",
                    "Jaga kesehatan dengan olahraga teratur",
                    "Lakukan pemeriksaan kesehatan rutin",
                    "Pastikan pola hidup sehat dan istirahat cukup"
                ],
                "color": "#10B981"
            },
            "Perawakan Tinggi": {
                "title": "✅ PERAWAKAN TINGGI",
                "actions": [
                    "Tinggi badan Anda di atas rata-rata",
                    "Pertahankan pola makan bergizi seimbang",
                    "Jaga kesehatan dengan olahraga teratur",
                    "Lakukan pemeriksaan kesehatan rutin",
                    "Pastikan pola hidup sehat"
                ],
                "color": "#10B981"
            }
        }
        
        return recommendations.get(status, recommendations["Normal"])
    
    def get_zscore_interpretation(self, zscore, is_adult=False):
        """
        Interpretasi nilai Z-Score untuk user
        """
        if is_adult:
            if zscore < -3:
                return "Tinggi badan berada sangat jauh di bawah standar rata-rata dewasa. Ini mungkin indikasi stunting yang terjadi di masa kecil."
            elif -3 <= zscore < -2:
                return "Tinggi badan berada di bawah standar rata-rata dewasa. Ini mungkin indikasi stunting yang terjah di masa kecil."
            elif -2 <= zscore <= 3:
                return "Tinggi badan berada dalam rentang normal untuk dewasa."
            else:
                return "Tinggi badan berada di atas standar rata-rata dewasa."
        else:
            if zscore < -3:
                return "Tinggi badan anak berada sangat jauh di bawah standar WHO untuk usianya."
            elif -3 <= zscore < -2:
                return "Tinggi badan anak berada di bawah standar WHO untuk usianya."
            elif -2 <= zscore <= 3:
                return "Tinggi badan anak sesuai dengan standar WHO untuk usianya."
            else:
                return "Tinggi badan anak berada di atas standar WHO untuk usianya."


# Testing function
if __name__ == "__main__":
    calculator = WHOZScoreCalculator()
    
    # Test case
    test_cases = [
        (12, 70, "laki-laki"),  # Stunted
        (24, 87, "laki-laki"),  # Normal
        (36, 85, "perempuan"),  # Severely Stunted
        (252, 165, "laki-laki"),  # Adult (21 tahun)
    ]
    
    for age, height, gender in test_cases:
        zscore, is_adult = calculator.calculate_zscore(age, height, gender)
        status = calculator.classify_nutrition_status(zscore, is_adult)
        age_display = f"{age} bulan ({age//12} tahun)" if age > 60 else f"{age} bulan"
        print(f"Umur: {age_display}, TB: {height} cm, Gender: {gender}")
        print(f"Z-Score: {zscore}, Status: {status}, Dewasa: {is_adult}\n")