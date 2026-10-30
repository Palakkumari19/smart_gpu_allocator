class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 10000;
    this.backoffMultiplier = 1.5;
    this.isExplicitDisconnect = false;
    this.onMessageCallback = null;
    this.onStatusCallback = null;
  }

  getWebSocketUrl() {
    if (typeof window === 'undefined') return 'ws://localhost:8000/ws';
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    const port = window.location.port !== '8000' ? '8000' : window.location.port;
    return `${protocol}//${host}:${port}/ws`;
  }

  connect(onMessage, onStatusChange) {
    this.onMessageCallback = onMessage;
    this.onStatusCallback = onStatusChange;
    this.isExplicitDisconnect = false;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    if (this.onStatusCallback) {
      this.onStatusCallback('Connecting...');
    }

    const url = this.getWebSocketUrl();

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.error('WebSocket initial connection failed:', err);
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log('WebSocket Connected to', url);
      this.reconnectDelay = 1000; // Reset backoff on successful connect
      if (this.onStatusCallback) {
        this.onStatusCallback('Connected');
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        // Handle both Bit 7 typed envelope ({ type, data }) and legacy top-level payload
        const data = (payload && payload.data) ? { ...payload.data, ...payload } : payload;
        if (this.onMessageCallback) {
          this.onMessageCallback(data, payload);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    this.ws.onerror = (error) => {
      console.warn('WebSocket encountered error:', error);
    };

    this.ws.onclose = (event) => {
      console.log(`WebSocket disconnected (code: ${event.code})`);
      if (this.onStatusCallback) {
        this.onStatusCallback('Disconnected');
      }
      if (!this.isExplicitDisconnect) {
        this.scheduleReconnect();
      }
    };
  }

  scheduleReconnect() {
    if (this.reconnectTimer || this.isExplicitDisconnect) return;

    const delay = Math.min(this.reconnectDelay, this.maxReconnectDelay);
    if (this.onStatusCallback) {
      this.onStatusCallback(`Reconnecting in ${(delay / 1000).toFixed(0)}s...`);
    }

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectDelay = Math.min(this.reconnectDelay * this.backoffMultiplier, this.maxReconnectDelay);
      if (!this.isExplicitDisconnect && this.onMessageCallback && this.onStatusCallback) {
        this.connect(this.onMessageCallback, this.onStatusCallback);
      }
    }, delay);
  }

  disconnect() {
    this.isExplicitDisconnect = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close(1000, 'Explicit client disconnect');
      } catch (e) {
        // Ignore close error
      }
      this.ws = null;
    }
    if (this.onStatusCallback) {
      this.onStatusCallback('Disconnected');
    }
  }
}

const wsService = new WebSocketService();
export default wsService;