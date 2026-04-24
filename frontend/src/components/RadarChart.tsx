import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const EMOTIONS = [
  { key: 'anger',        label: 'Anger',        color: '#FF6B6B' },
  { key: 'anticipation', label: 'Anticipation',  color: '#FFD166' },
  { key: 'disgust',      label: 'Disgust',       color: '#9B72CF' },
  { key: 'fear',         label: 'Fear',          color: '#6B7BFF' },
  { key: 'joy',          label: 'Joy',           color: '#06D6A0' },
  { key: 'sadness',      label: 'Sadness',       color: '#4D9FFF' },
  { key: 'surprise',     label: 'Surprise',      color: '#F4A261' },
  { key: 'trust',        label: 'Trust',         color: '#43C59E' },
];

interface EmotionBarChartProps {
  emotionVector: Record<string, number>;
  dominantEmotion: string;
}

export default function EmotionBarChart({ emotionVector, dominantEmotion }: EmotionBarChartProps) {
  const values  = EMOTIONS.map((e) => +(emotionVector[e.key] ?? 0).toFixed(3));
  const colors  = EMOTIONS.map((e) =>
    e.key === dominantEmotion.toLowerCase()
      ? e.color                          // full opacity for dominant
      : e.color + 'BB'                   // ~73% opacity for others
  );
  const borders = EMOTIONS.map((e) =>
    e.key === dominantEmotion.toLowerCase() ? e.color : e.color + '55'
  );

  const data = {
    labels: EMOTIONS.map((e) => e.label),
    datasets: [
      {
        label: 'NRC Score',
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: { parsed?: { y?: number } }) =>
            ` ${((ctx.parsed?.y ?? 0) * 100).toFixed(1)}%`,
        },
        backgroundColor: 'rgba(0,0,0,0.85)',
        titleFont: { family: "'Inter', sans-serif", size: 12 },
        bodyFont:  { family: "'Inter', sans-serif", size: 12 },
        padding: 10,
        cornerRadius: 8,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          font: { family: "'Inter', sans-serif", size: 11 },
          color: (ctx: { index?: number }) => {
            const key = EMOTIONS[ctx.index ?? 0]?.key ?? '';
            return key === dominantEmotion.toLowerCase()
              ? '#000'
              : 'rgba(111,111,111,0.75)';
          },
        },
        border: { display: false },
      },
      y: {
        min: 0,
        max: Math.max(...values, 0.5),
        grid: { color: 'rgba(111,111,111,0.1)' },
        ticks: {
          font: { family: "'Inter', sans-serif", size: 10 },
          color: 'rgba(111,111,111,0.6)',
          callback: (v: number | string) => `${(+v * 100).toFixed(0)}%`,
        },
        border: { display: false },
      },
    },
  };

  return (
    <div style={{ width: '100%' }}>
      <Bar data={data} options={options as Parameters<typeof Bar>[0]['options']} />
    </div>
  );
}
