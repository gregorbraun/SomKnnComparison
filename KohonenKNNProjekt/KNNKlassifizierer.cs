// Implementiert den K-Nearest-Neighbor (KNN) Algorithmus zur Klassenvorhersage basierend auf Trainingsdaten.

using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenKNNProjekt
{
    public static class KNNKlassifizierer
    {
        public static string Vorhersagen(List<Datenpunkt> trainingsDaten, double[] eingabe, int k)
        {
            return trainingsDaten
                .OrderBy(p => EuklidischerAbstand(p.Merkmale, eingabe))
                .Take(k)
                .GroupBy(p => p.Klasse)
                .OrderByDescending(g => g.Count())
                .First().Key;
        }

        private static double EuklidischerAbstand(double[] a, double[] b)
        {
            return Math.Sqrt(a.Select((v, i) => Math.Pow(v - b[i], 2)).Sum());
        }
    }
}
