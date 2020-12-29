using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_01
{
    class Layer
    {
        public List<Neuron> Neurons = new List<Neuron>();
        public int Length => Neurons.Count;
        public void Add(Neuron neuron) => Neurons.Add(neuron);
        public void Remove(Neuron neuron) => Neurons.Remove(neuron);

        static System.Random Random = new System.Random();

        //Sets the values of Neurons to a new set of values
        public void SetValues(float[] NewValues)
        {
            for (int i = 0; i < Length; i++)
            {
                this[i].Value = NewValues[i];
            }
        }

        public Neuron this[int index]
        {
            get => Neurons[index];
            set => Neurons[index] = value;
        }
        //Sets the number of weights in previous layer neurons to number of neurons in this layer
        public void AppendToLayer(Layer Prev)
        {
            foreach (Neuron neuron in Prev.Neurons)
            {
                neuron.Weights = new float[this.Length];
            }
        }
        //Returns a random float between 0.0 and 1.0
        public static float GetRandomFloat()
        {
            return (float)Random.Next(-100, 100) / 100f;
        }
        //Sets all Neuron Biases and Weights to Random Floats
        public void Randomize()
        {
            foreach (Neuron neuron in this.Neurons)
            {
                if (neuron.Weights != null)
                {
                    neuron.Bias = GetRandomFloat();
                    for (int i = 0; i < neuron.Weights.Length; i++)
                    {
                        neuron.Weights[i] = GetRandomFloat();
                    }
                }
                else
                {
                    neuron.Bias = 0f;
                }
            }
            
        }
        public Layer(int Count)
        {
            // Adds empty neurons to layer based on Count
            for (int i = 0; i < Count; i++)
            {
                Neuron NewNeuron = new Neuron();
                this.Add(NewNeuron);
            }
        }
    }
}
