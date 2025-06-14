// Projekt C# - Kohonen-Karte mit KNN-Vergleich
// Verwendeter Datensatz: Iris (150 Stichproben, 4 Merkmale, 3 Klassen)

using KohonenKNNProjekt;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace KohonenKNN
{
    class Program
    {
        static void Main(string[] args)
        {
            var daten = LadeIrisDatensatz("iris.csv");
            var som = new SOM(10, 10, 4);
            som.Trainieren(daten.Select(d => d.Merkmale).ToList(), 1000);

            Console.WriteLine("KNN vs SOM - Klassenvorhersage:\n");

            int k = 3;
            int korrektKNN = 0, korrektSOM = 0;
            foreach (var stichprobe in daten)
            {
                var vorhergesagtKNN = KNNKlassifizierer.Vorhersagen(daten, stichprobe.Merkmale, k);
                var vorhergesagtSOM = som.Klassifizieren(stichprobe.Merkmale, daten);

                if (vorhergesagtKNN == stichprobe.Klasse) korrektKNN++;
                if (vorhergesagtSOM == stichprobe.Klasse) korrektSOM++;
            }

            Console.WriteLine($"Genauigkeit KNN: {(double)korrektKNN / daten.Count * 100:0.00}%");
            Console.WriteLine($"Genauigkeit SOM: {(double)korrektSOM / daten.Count * 100:0.00}%");
        }

        // Liest den Iris-Datensatz ein
        static List<Datenpunkt> LadeIrisDatensatz(string pfad)
        {
            var zeilen = IrisData.Data.Split("\n".ToCharArray());//File.ReadAllLines(pfad).Skip(1); // Ignore la 1re ligne (en-tête)
            var datensatz = new List<Datenpunkt>();
            foreach (var zeile in zeilen)
            {
                var teile = zeile.Split(',');
                var merkmale = teile.Take(4).Select(p => double.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                var klasse = teile[4];
                datensatz.Add(new Datenpunkt { Merkmale = merkmale, Klasse = klasse });
            }
            return datensatz;
        }

    }

    // Repräsentiert einen Datenpunkt mit Merkmalen und Klasse
    class Datenpunkt
    {
        public double[] Merkmale = Array.Empty<double>();
        public string Klasse = "";
    }

    // Ein einzelner Neuron auf der Karte
    class Neuron
    {
        public double[] Gewichte;
        public int X, Y;

        public Neuron(int merkmalAnzahl, int x, int y)
        {
            X = x; Y = y;
            Gewichte = new double[merkmalAnzahl];
            var zufall = new Random();
            for (int i = 0; i < merkmalAnzahl; i++)
                Gewichte[i] = zufall.NextDouble();
        }

        public double AbstandZu(double[] eingabe)
        {
            return Math.Sqrt(Gewichte.Select((w, i) => Math.Pow(w - eingabe[i], 2)).Sum());
        }
    }

    // Selbstorganisierende Karte (SOM)
    class SOM
    {
        private Neuron[,] karte;
        private int breite, hoehe, merkmalAnzahl;
        private Random zufall = new Random();

        public SOM(int breite, int hoehe, int merkmalAnzahl)
        {
            this.breite = breite;
            this.hoehe = hoehe;
            this.merkmalAnzahl = merkmalAnzahl;
            karte = new Neuron[breite, hoehe];
            for (int x = 0; x < breite; x++)
                for (int y = 0; y < hoehe; y++)
                    karte[x, y] = new Neuron(merkmalAnzahl, x, y);
        }

        // Training der SOM mit Eingabedaten
        public void Trainieren(List<double[]> daten, int iterationen)
        {
            for (int iter = 0; iter < iterationen; iter++)
            {
                var eingabe = daten[zufall.Next(daten.Count)];
                var bmu = FindeBMU(eingabe);

                double lernrate = 0.1 * (1.0 - (double)iter / iterationen);
                int radius = (int)(Math.Max(breite, hoehe) * (1.0 - (double)iter / iterationen));

                for (int x = 0; x < breite; x++)
                {
                    for (int y = 0; y < hoehe; y++)
                    {
                        var neuron = karte[x, y];
                        double dist = Math.Sqrt(Math.Pow(x - bmu.X, 2) + Math.Pow(y - bmu.Y, 2));
                        if (dist <= radius)
                        {
                            for (int i = 0; i < merkmalAnzahl; i++)
                            {
                                neuron.Gewichte[i] += lernrate * (eingabe[i] - neuron.Gewichte[i]);
                            }
                        }
                    }
                }
            }
        }

        // Findet die Beste passende Einheit (BMU)
        private Neuron FindeBMU(double[] eingabe)
        {
            Neuron beste = karte[0, 0];
            double minDistanz = beste.AbstandZu(eingabe);
            foreach (var neuron in karte)
            {
                double dist = neuron.AbstandZu(eingabe);
                if (dist < minDistanz)
                {
                    minDistanz = dist;
                    beste = neuron;
                }
            }
            return beste;
        }

        // Klassifiziert einen Eingabewert basierend auf SOM und Trainingsdaten
        public string Klassifizieren(double[] eingabe, List<Datenpunkt> trainingsDaten)
        {
            var bmu = FindeBMU(eingabe);
            var naechste = trainingsDaten.OrderBy(dp => bmu.AbstandZu(dp.Merkmale)).Take(5);
            return naechste.GroupBy(dp => dp.Klasse).OrderByDescending(g => g.Count()).First().Key;
        }
    }

    // KNN-Klassifizierer
    static class KNNKlassifizierer
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
