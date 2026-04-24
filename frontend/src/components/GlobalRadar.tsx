import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

interface GlobalRadarProps {
  emotionVector: Record<string, number>;
}

export default function GlobalRadar({ emotionVector }: GlobalRadarProps) {
  const EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'];
  
  const data = {
    labels: EMOTIONS.map(e => e.toUpperCase()),
    datasets: [
      {
        label: 'Global Affective Average',
        data: EMOTIONS.map(e => +(emotionVector[e] ?? 0).toFixed(3)),
        backgroundColor: 'rgba(111, 111, 111, 0.4)', // glowing #6F6F6F fill approx
        borderColor: '#ffffff', // thin white lines
        borderWidth: 1,
        pointBackgroundColor: '#ffffff',
        pointBorderColor: '#ffffff',
        pointHoverBackgroundColor: '#ffffff',
        pointHoverBorderColor: '#ffffff',
        pointRadius: 2,
        fill: true,
      },
    ],
  };

  const options = {
    scales: {
      r: {
        angleLines: {
          color: 'rgba(111, 111, 111, 0.3)',
        },
        grid: {
          color: 'rgba(111, 111, 111, 0.3)',
        },
        pointLabels: {
          color: '#ffffff',
          font: {
            family: "'Inter', sans-serif",
            size: 11,
            letterSpacing: '0.1em'
          }
        },
        ticks: {
          display: false, // hide the scale numbers on radar
        }
      }
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0,0,0,0.85)',
        titleFont: { family: "'Inter', sans-serif", size: 12 },
        bodyFont:  { family: "'Inter', sans-serif", size: 12 },
        padding: 10,
        cornerRadius: 8,
      }
    }
  };

  return <Radar data={data} options={options} />;
}
