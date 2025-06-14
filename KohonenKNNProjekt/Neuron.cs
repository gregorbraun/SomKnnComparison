// Definiert ein einzelnes Neuron innerhalb der SOM-Karte. Speichert Position und Gewichte.

using System;
using System.Linq;

namespace KohonenKNNProjekt
{
    public class Neuron
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
}
