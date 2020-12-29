using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_01
{
    class Neuron
    {
        public float Value { get; set; }
        public float Bias { get; set; }
        public float[] Weights { get; set; }
        public float Gamma { get; set; }
        public Neuron()
        {
            Value = 0;
            Bias = 0;
        }
    }
}
