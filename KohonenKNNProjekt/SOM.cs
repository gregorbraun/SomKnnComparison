// Enthält die selbstorganisierende Karte (SOM) zur unüberwachten Klassifikation,
// zur Zuordnung von Klassenlabels zu Neuronen und zur 2D-Visualisierung in der Konsole.

using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenKNNProjekt
{
    public class SOM
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

        // Trainiert die SOM mit den gegebenen Daten für eine bestimmte Anzahl von Iterationen
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
                                neuron.Gewichte[i] += lernrate * (eingabe[i] - neuron.Gewichte[i]);
                        }
                    }
                }
            }
        }

        // Findet die Best Matching Unit (BMU) für einen gegebenen Eingabevektor
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

        // Klassifiziert einen Eingabevektor basierend auf Trainingsdaten (unsupervised mapping)
        public string Klassifizieren(double[] eingabe, List<Datenpunkt> trainingsDaten)
        {
            var bmu = FindeBMU(eingabe);
            var naechste = trainingsDaten.OrderBy(dp => bmu.AbstandZu(dp.Merkmale)).Take(5);
            return naechste.GroupBy(dp => dp.Klasse).OrderByDescending(g => g.Count()).First().Key;
        }

        // Ordnet jedem Neuron ein Klassenlabel zu, basierend auf den nächstgelegenen Trainingsdatenpunkten
        public Dictionary<(int x, int y), string> LabelNeuronen(List<Datenpunkt> daten)
        {
            var labels = new Dictionary<(int, int), List<string>>();

            foreach (var punkt in daten)
            {
                var bmu = FindeBMU(punkt.Merkmale);
                var key = (bmu.X, bmu.Y);
                if (!labels.ContainsKey(key))
                    labels[key] = new List<string>();
                labels[key].Add(punkt.Klasse);
            }

            var neuronLabels = new Dictionary<(int, int), string>();
            foreach (var eintrag in labels)
            {
                var haeufigste = eintrag.Value
                    .GroupBy(k => k)
                    .OrderByDescending(g => g.Count())
                    .First().Key;

                neuronLabels[eintrag.Key] = haeufigste;
            }

            return neuronLabels;
        }

        // Gibt eine einfache 2D-Visualisierung der Karte in der Konsole aus
        public void VisualisiereKarte(Dictionary<(int x, int y), string> neuronLabels)
        {
            Console.WriteLine("\nKohonen-Karte (2D Visualisierung):\n");
            for (int y = 0; y < hoehe; y++)
            {
                for (int x = 0; x < breite; x++)
                {
                    var key = (x, y);
                    if (neuronLabels.ContainsKey(key))
                    {
                        string label = neuronLabels[key];
                        Console.Write($"{LabelKurz(label)} ");
                    }
                    else
                    {
                        Console.Write(". ");
                    }
                }
                Console.WriteLine();
            }
        }

        // Kürzt den Klassennamen für die Anzeige (z.B. "setosa" → "S")
        private string LabelKurz(string klasse)
        {
            return klasse switch
            {
                "setosa" => "S",
                "versicolor" => "V",
                "virginica" => "I",
                _ => "?"
            };
        }
    }
}
