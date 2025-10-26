import sqlite3
import json
import traceback
import sys
import os

# ensure project root is on sys.path so 'agent' package can be imported when running
# this script from the tests/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.return_agent import ReturnAgent


def get_sample_ids():
    conn = sqlite3.connect("ecomarket.db")
    cur = conn.cursor()
    try:
        cur.execute("SELECT tracking FROM orders LIMIT 1")
        r = cur.fetchone()
        tracking = r[0] if r else None
        cur.execute("SELECT sku FROM products LIMIT 1")
        s = cur.fetchone()
        sku = s[0] if s else None
    finally:
        conn.close()
    return tracking, sku


if __name__ == '__main__':
    print('Running agent tools smoke test')
    tracking, sku = None, None
    try:
        tracking, sku = get_sample_ids()
        print(f"Sample tracking: {tracking}")
        print(f"Sample sku: {sku}")
    except Exception as e:
        print('Error reading sample ids from DB:')
        traceback.print_exc()

    try:
        print('\nInstantiating ReturnAgent...')
        ra = ReturnAgent()
        print('ReturnAgent initialized')
    except Exception as e:
        print('Error initializing ReturnAgent:')
        traceback.print_exc()
        ra = None

    if ra:
        try:
            t = tracking or 'TEST_TRACK'
            s = sku or 'TEST_SKU'
            print('\n--- _consultar_estado_pedido ---')
            out1 = ra._consultar_estado_pedido(t)
            print(out1)

            print('\n--- _verificar_elegibilidad_producto ---')
            out2 = ra._verificar_elegibilidad_producto(t, s)
            print(out2)

            print('\n--- _consultar_politicas ---')
            out3 = ra._consultar_politicas('general')
            print(out3)

            print('\n--- _generar_etiqueta_devolucion ---')
            out4 = ra._generar_etiqueta_devolucion(t, s, carrier='EcoShip')
            print(out4)

            print('\n--- Example agent.run (short prompt asking to check order) ---')
            prompt = f"Por favor verifica el estado del pedido con tracking {t} y resume si es elegible para devolución para el sku {s}. Responde en español."
            try:
                res = ra.run(prompt)
                print('\nAgent.run result:\n')
                print(res)
            except Exception:
                print('agent.run raised an exception:')
                traceback.print_exc()

        except Exception as e:
            print('Error calling tool methods:')
            traceback.print_exc()
    else:
        print('Skipping tool calls because ReturnAgent failed to initialize.')
