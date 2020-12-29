using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_01
{
    class Data
    {
        public float[] Values { get; set; }
        public int Length => Values.Length;
        public float this[int index]
        {
            get => Values[index];
            set => Values[index] = value;
        }
        public float IsBlueBerry { get; set; }
    }
}
