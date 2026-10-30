from fastapi.testclient import TestClient
import backend.server as server

client = TestClient(server.app)


def test_scheduler_persistence_through_start_and_reset():
    schedulers = ['fcfs', 'round_robin', 'least_loaded', 'rl', 'marl']
    for sched in schedulers:
        # set scheduler
        r = client.post('/set-scheduler', json={'scheduler': sched})
        assert r.status_code == 200
        j = r.json()
        assert j.get('status') == 'ok'
        assert j.get('scheduler') == sched

        # check dashboard-state reflects it
        r = client.get('/dashboard-state')
        assert r.status_code == 200
        ds = r.json()
        assert ds.get('scheduler') == sched

        # start simulation
        r = client.post('/start')
        assert r.status_code == 200

        # after starting, scheduler should still be same
        r = client.get('/dashboard-state')
        assert r.status_code == 200
        ds = r.json()
        assert ds.get('scheduler') == sched

        # reset simulation and ensure scheduler preserved
        r = client.post('/reset')
        assert r.status_code == 200
        r = client.get('/dashboard-state')
        assert r.status_code == 200
        ds = r.json()
        assert ds.get('scheduler') == sched
