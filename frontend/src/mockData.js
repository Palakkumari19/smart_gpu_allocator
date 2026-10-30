export const initialGpus = [
  { id: 0, status: 'Healthy', memory: 62, temperature: 58, queue: 2, running: 1, currentTask: 'LLM-104', reward: 8.4, utilization: 68 },
  { id: 1, status: 'High Load', memory: 81, temperature: 73, queue: 4, running: 2, currentTask: 'Video-221', reward: 5.9, utilization: 84 },
  { id: 2, status: 'Healthy', memory: 48, temperature: 51, queue: 1, running: 1, currentTask: 'Image-318', reward: 7.8, utilization: 55 },
  { id: 3, status: 'Idle', memory: 22, temperature: 35, queue: 0, running: 0, currentTask: '-', reward: 6.2, utilization: 30 },
];

export const initialTasks = [
  { id: 'T-101', type: 'LLM', memory: 6, duration: 5, priority: 'High' },
  { id: 'T-102', type: 'Image', memory: 2, duration: 2, priority: 'Medium' },
  { id: 'T-103', type: 'Video', memory: 4, duration: 4, priority: 'High' },
  { id: 'T-104', type: 'LLM', memory: 7, duration: 6, priority: 'Critical' },
];

export const initialEvents = [
  { id: 1, time: '00:01:12', message: 'Task generated', type: 'info' },
  { id: 2, time: '00:01:14', message: 'GPU 2 accepted task', type: 'success' },
  { id: 3, time: '00:01:18', message: 'Temperature increased', type: 'warning' },
];

export const metricSeries = [
  { name: 'T1', completed: 4, latency: 18, reward: 5.1 },
  { name: 'T2', completed: 7, latency: 22, reward: 6.2 },
  { name: 'T3', completed: 10, latency: 20, reward: 6.8 },
  { name: 'T4', completed: 14, latency: 24, reward: 7.1 },
  { name: 'T5', completed: 18, latency: 21, reward: 7.9 },
];

export const schedulerComparison = [
  { name: 'MARL', completed: 92, latency: 68, utilization: 82 },
  { name: 'Round Robin', completed: 46, latency: 88, utilization: 63 },
  { name: 'Least Loaded', completed: 50, latency: 74, utilization: 71 },
];
