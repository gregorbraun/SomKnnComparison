// Diese Datei definiert eine Hilfsklasse zur Erstellung und Anzeige einer Konfusionsmatrix
// für Klassifikationsaufgaben mit mehreren Klassen.

using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenKNNProjekt
{
    public class ConfusionMatrix
    {
        private readonly Dictionary<string, int> labelIndices;
        private readonly string[] labels;
        private readonly int[,] matrix;

        public ConfusionMatrix(List<string> echteKlassen, List<string> vorhergesagteKlassen)
        {
            if (echteKlassen.Count != vorhergesagteKlassen.Count)
                throw new ArgumentException("Listen müssen gleich lang sein");

            labels = echteKlassen.Concat(vorhergesagteKlassen).Distinct().OrderBy(l => l).ToArray();
            labelIndices = labels.Select((label, index) => new { label, index })
                                 .ToDictionary(x => x.label, x => x.index);

            matrix = new int[labels.Length, labels.Length];

            for (int i = 0; i < echteKlassen.Count; i++)
            {
                int actualIndex = labelIndices[echteKlassen[i]];
                int predictedIndex = labelIndices[vorhergesagteKlassen[i]];
                matrix[actualIndex, predictedIndex]++;
            }
        }

        public void Drucken(string titel)
        {
            Console.WriteLine($"\nKonfusionsmatrix – {titel}");
            Console.Write("".PadRight(12));
            foreach (var label in labels)
                Console.Write(label.PadRight(12));
            Console.WriteLine();

            for (int i = 0; i < labels.Length; i++)
            {
                Console.Write(labels[i].PadRight(12));
                for (int j = 0; j < labels.Length; j++)
                {
                    Console.Write(matrix[i, j].ToString().PadRight(12));
                }
                Console.WriteLine();
            }
        }
    }
}
