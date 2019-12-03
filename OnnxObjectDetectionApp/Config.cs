using System;
using System.Collections.Generic;
using System.Text;

namespace OnnxObjectDetectionApp
{
    public class Box
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float W { get; set; }
        public float H { get; set; }
    }
    public class OscCom
    {
        public string Ip { get; set; }
        public int Port { get; set; }
    }
    public class ConfigParameters
    {
        public ICollection<Box> Quadrants { get; set; }
        public OscCom OscCom { get; set; }
        public int TimeToStationary { get; set; }

    }
}
