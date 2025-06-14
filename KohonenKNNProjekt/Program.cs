// Hauptprogramm zur Ausführung des Vergleichs zwischen SOM und KNN mit Konfusionsmatrix-Ausgabe

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using KohonenKNNProjekt;

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

            var echteKlassen = new List<string>();
            var vorhersagenKNN = new List<string>();
            var vorhersagenSOM = new List<string>();

            foreach (var stichprobe in daten)
            {
                var vorhergesagtKNN = KNNKlassifizierer.Vorhersagen(daten, stichprobe.Merkmale, k);
                var vorhergesagtSOM = som.Klassifizieren(stichprobe.Merkmale, daten);

                if (vorhergesagtKNN == stichprobe.Klasse) korrektKNN++;
                if (vorhergesagtSOM == stichprobe.Klasse) korrektSOM++;

                echteKlassen.Add(stichprobe.Klasse);
                vorhersagenKNN.Add(vorhergesagtKNN);
                vorhersagenSOM.Add(vorhergesagtSOM);
            }

            Console.WriteLine($"Genauigkeit KNN: {(double)korrektKNN / daten.Count * 100:0.00}%");
            Console.WriteLine($"Genauigkeit SOM: {(double)korrektSOM / daten.Count * 100:0.00}%");

            // Konfusionsmatrizen anzeigen
            var matrixKNN = new ConfusionMatrix(echteKlassen, vorhersagenKNN);
            var matrixSOM = new ConfusionMatrix(echteKlassen, vorhersagenSOM);

            matrixKNN.Drucken("KNN");
            matrixSOM.Drucken("SOM");

            var somLabels = som.LabelNeuronen(daten);
            som.VisualisiereKarte(somLabels);


        }

        // Liest den Iris-Datensatz ein
        static List<Datenpunkt> LadeIrisDatensatz(string pfad)
        {
            var zeilen = IrisData.Data.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            var datensatz = new List<Datenpunkt>();

            foreach (var zeile in zeilen)
            {
                var teile = zeile.Split(',');
                var merkmale = teile.Take(4)
                    .Select(p => double.Parse(p, CultureInfo.InvariantCulture))
                    .ToArray();
                var klasse = teile[4];
                datensatz.Add(new Datenpunkt { Merkmale = merkmale, Klasse = klasse });
            }

            return datensatz;
        }
    }
}
