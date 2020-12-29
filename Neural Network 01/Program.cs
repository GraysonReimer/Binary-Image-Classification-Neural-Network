using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Web.Script.Serialization;

namespace Neural_Network_01
{
    static class Program
    {
        public static int ImageResolution = 16;
        //List of Layers that make up the Neural Network
        public static Layer[] Layers = new Layer[] { new Layer(ImageResolution * ImageResolution), new Layer(172), new Layer(172), new Layer(2) };
        //Easy Access to the first layer of the network
        public static Layer Inputs => Layers[0];

        //Easy Access to the last layer of the network
        public static Layer Outputs => Layers[Layers.Length - 1];

        //The data that will be fed into the Network
        public static Data[] TrainingData;

        //The data that will asess the Network
        public static Data[] TestingData;

        public static Data data;

        public static int TrainingDataIndex = 0;
        public static int TestingDataIndex = 0;

        public static System.Random random = new System.Random();

        //Appends each layer to the previous layer and then randomizes the layers
        static void InitializeLayers()
        {
            Console.WriteLine("\n\nInitializing Layers...");
            if (Layers.Length >= 3)
            {
                for (int i = 1; i < Layers.Length; i++)
                {
                    Layers[i].AppendToLayer(Layers[i - 1]);
                }
                for (int i = 0; i < Layers.Length; i++)
                {
                    Layers[i].Randomize();
                }
            }
        }


        static void InitializeData()
        {
            Console.WriteLine("\n\nInitializing Data...");
            string BlueberryDirectory = "C:/Users/Grays PC/source/repos/Neural Network 01/Neural Network 01/Image Data/Blueberries";
            string StrawberryDirectory = "C:/Users/Grays PC/source/repos/Neural Network 01/Neural Network 01/Image Data/Strawberries";
            List<Data> STImageData = new List<Data>();
            List<Data> BLImageData = new List<Data>();
            List<Data> NewTrainingImageData = new List<Data>();
            List<Data> NewTestingImageData = new List<Data>();
            foreach (string BLimage in Directory.GetFiles(BlueberryDirectory))
            {
                List<float> PixelValues = new List<float>();
                Bitmap image = new Bitmap(Image.FromFile(BLimage), newSize: new Size(ImageResolution, ImageResolution));
                Data newData = new Data();
                for (int i = 0; i < ImageResolution; i++)
                {
                    for (int j = 0; j < ImageResolution; j++)
                    {
                        float ColorValue = image.GetPixel(i, j).R;
                        PixelValues.Add(Remap(ColorValue, 0, 255, 0, 1));
                    }
                }
                newData.IsBlueBerry = 1;
                newData.Values = PixelValues.ToArray();
                BLImageData.Add(newData);
            }
            foreach (string STimage in Directory.GetFiles(StrawberryDirectory))
            {
                List<float> PixelValues = new List<float>();
                Bitmap image = new Bitmap(STimage);
                Data newData = new Data();
                for (int i = 0; i < ImageResolution; i++)
                {
                    for (int j = 0; j < ImageResolution; j++)
                    {
                        float ColorValue = image.GetPixel(i, j).R;
                        PixelValues.Add(Remap(ColorValue, 0, 255, 0, 1));
                    }
                }
                newData.IsBlueBerry = 0;
                newData.Values = PixelValues.ToArray();
                STImageData.Add(newData);
            }
            int BLSeparationNumber = Convert.ToInt32((float)BLImageData.Count * 0.26f);
            int STSeparationNumber = Convert.ToInt32((float)STImageData.Count * 0.26f);
            for (int i = 0; i < STImageData.Count; i++)
            {
                if (i < STSeparationNumber)
                {
                    NewTestingImageData.Add(STImageData[i]);
                }
                else
                {
                    NewTrainingImageData.Add(STImageData[i]);
                }
            }
            for (int i = 0; i < BLImageData.Count; i++)
            {
                if (i < BLSeparationNumber)
                {
                    NewTestingImageData.Add(BLImageData[i]);
                }
                else
                {
                    NewTrainingImageData.Add(BLImageData[i]);
                }
            }
            TrainingData = NewTrainingImageData.ToArray();
            TestingData = NewTestingImageData.ToArray();
            RandomizeTrainingData();
            RandomizeTestingData();
        }
        public static float Remap(this float from, float fromMin, float fromMax, float toMin, float toMax)
        {
            var fromAbs = from - fromMin;
            var fromMaxAbs = fromMax - fromMin;

            var normal = fromAbs / fromMaxAbs;

            var toMaxAbs = toMax - toMin;
            var toAbs = toMaxAbs * normal;

            var to = toAbs + toMin;

            return to;
        }
        static void RandomizeTrainingData()
        {
            TrainingData = TrainingData.ToList().OrderBy(x => random.Next(0, 100000)).ToArray();
        }
        static void RandomizeTestingData()
        {
            TestingData = TestingData.ToList().OrderBy(x => random.Next(0, 100000)).ToArray();
        }
        static void GetNextTrainingData()
        {
            data = TrainingData[TrainingDataIndex];
            TrainingDataIndex++;
            if (TrainingDataIndex == TrainingData.Count())
            {
                TrainingDataIndex = 0;
                RandomizeTrainingData();
            }
        }
        static void GetNextTestingData()
        {
            data = TestingData[TestingDataIndex];
            TestingDataIndex++;
            if (TestingDataIndex == TestingData.Count())
            {
                TestingDataIndex = 0;
                RandomizeTestingData();
            }
        }
        //Returns the Sigmoid of any inputed float
        public static float Sigmoid(float x)
        {
            return (float)(1.0d / (1.0d + Math.Exp(-x)));
        }

        //Feed Data into the Network to get an output
        static void FeedForward()
        {
            Inputs.SetValues(data.Values);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layer layer = Layers[i];
                int index = 0;
                foreach (Neuron neuron in layer.Neurons)
                {
                    float newValue = 0;
                    foreach (Neuron prevNeuron in Layers[i-1].Neurons)
                    {
                        newValue += prevNeuron.Value * prevNeuron.Weights[index];
                        if (Double.IsNaN(Sigmoid((newValue+neuron.Bias))))
                        {
                            throw new Exception("NaN value was found in network.");// LearnRate 0.5, BiasLeranRate 0.1 = 1:20s : NaN; LearnRate 0.005, BiasLearningRate 0.001 = 4:31s : 61.8% no ch;
                        }
                    }
                    newValue += neuron.Bias;
                    neuron.Value = Sigmoid(newValue);
                    index++;
                }
            }
        }
        //Adjust the biases and weights
        static void BackPropagate()
        {
            float LearningRate = 0.005f;
            float BiasLearningRate = 0.001f;
            float[] expectedAnswer;
            if (data.IsBlueBerry == 1)
                expectedAnswer = new float[] { 1, 0 };
            else
                expectedAnswer = new float[] { 0, 1 };


            //Begin with changing the Weights connected to the output layer

            //foreach Neuron in Outputs
            for (int NeuronNum = 0; NeuronNum < Outputs.Length; NeuronNum++)
            {
                Neuron neuron = Outputs[NeuronNum];
                float Gamma = (neuron.Value - expectedAnswer[NeuronNum]) * (1 - (neuron.Value) * (neuron.Value));
                neuron.Gamma = Gamma;
                //foreach Neuron in Previous Layer
                for (int PrevLayerNum = 0; PrevLayerNum < Layers[Layers.Length - 2].Length; PrevLayerNum++)
                {
                    //Calculate a New Value to Subtract
                    float Delta = Gamma*Layers[Layers.Length - 2][PrevLayerNum].Value;
                    //Apply the New Value to Adjust the Weight thats pointing to the neuron
                    Layers[Layers.Length - 2][PrevLayerNum].Weights[NeuronNum] -= Delta * LearningRate;
                    //Apply the New Value to Adjust the Bias of the Previous Weight
                    Layers[Layers.Length - 2][PrevLayerNum].Bias -= Delta * BiasLearningRate;
                }
            }
            
            for (int LayerNum = Layers.Length - 2; LayerNum > 0; LayerNum--)
            {
                Layer layer = Layers[LayerNum];

                for (int NeuronNum = 0; NeuronNum < layer.Length; NeuronNum++)
                {
                    Neuron neuron = layer[NeuronNum];
                    float Gamma = 0;
                    for (int NeuronNum2 = 0; NeuronNum2 < Layers[LayerNum + 1].Length; NeuronNum2++)
                    {
                        Gamma += Layers[LayerNum + 1][NeuronNum2].Gamma * Layers[LayerNum][NeuronNum].Weights[NeuronNum2];
                    }
                    Gamma *= (1 - neuron.Value * neuron.Value);
                    neuron.Gamma = Gamma;
                    //foreach Neuron in Previous Layer
                    for (int PrevLayerNum = 0; PrevLayerNum < Layers[LayerNum-1].Length; PrevLayerNum++)
                    {
                        //Calculate a New Value to Subtract
                        float Delta = Gamma * Layers[LayerNum - 1][PrevLayerNum].Value;
                        //Apply the New Value to Adjust the Weight thats pointing to the neuron
                        Layers[LayerNum - 1][PrevLayerNum].Weights[NeuronNum] -= Delta * LearningRate;
                        //Apply the New Value to Adjust the Bias of the Previous Weight
                        Layers[LayerNum - 1][PrevLayerNum].Bias -= Delta * BiasLearningRate;
                    }
                }
            }

        }

        //Neuron 0 is for answer of BlueBerries. Neuron 1 is for answer of StrawBerries.

        //Feeds Forward and then back propagates for the specified number of iterations
        static void RunNetwork(int Iterations, bool log)
        {
            Console.WriteLine("\n\nRunning Network...\n\n");
            for (int i = 0; i < Iterations; i++)
            {
                print($"Iteration {i+1} / {Iterations} Completed\n");
                GetNextTrainingData();
                FeedForward();
                BackPropagate();
            }

            void print(string str)
            {
                if (log)
                {
                    Console.WriteLine(str);
                }
            }
        }
        static void TestNetwork(int Iterations)
        {
            int CorrectAnswers = 0;
            for (int i = 0; i < Iterations; i++)
            {
                GetNextTestingData();
                FeedForward();
                Console.WriteLine("\n\nImage: " + data.IsBlueBerry);
                Console.WriteLine("0: " + Outputs[0].Value);
                Console.WriteLine("1: " + Outputs[1].Value);
                if ((data.IsBlueBerry == 1 && Outputs[0].Value > Outputs[1].Value) || (data.IsBlueBerry == 0 && Outputs[1].Value > Outputs[0].Value))
                {
                    Console.WriteLine("\nCorrect");
                    CorrectAnswers++;
                }
                else
                {
                    Console.WriteLine("\nIncorrect");
                }
            }
            Console.WriteLine($"\n\nNetwork Accuracy: {(float)CorrectAnswers/(float)Iterations*100f}%");

        }
        static void Main(string[] args)
        {
            var jsSerializer = new JavaScriptSerializer();
            jsSerializer.MaxJsonLength = Int32.MaxValue;
            InitializeData();
            InitializeLayers();
            RunNetwork(1250, false);
            TestNetwork(1000);
            Console.ReadLine();
        }
    }
}
