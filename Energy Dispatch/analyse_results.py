import numpy as np
import os

if __name__ == '__main__':
    # consider 10, 30 and 60 min frequency
    for SHIFT in [1,3,6]:
        print('-'*20)
        print(f'SHIFT: {SHIFT}')
        print('-'*20)

        baseline_erm = np.load('results/baseline_erm.npy')
        baseline_last10 = np.load(f'results/baseline_last10_shift{SHIFT}.npy')
        baseline_oracle = np.load(f'results/baseline_oracle_shift{SHIFT}.npy')
        print('Oracle: '+str(baseline_oracle.mean().round(3)))
        print('ERM: '+str(baseline_erm.mean().round(3)))
        print('Stationary: '+str(baseline_last10.mean().round(3)))

        e2e = {
            'cal' : {
                'hist' : [np.load('results/e2e/cal/historical/'+f) for f in os.listdir('results/e2e/cal/historical/') if f'{SHIFT}.npy' in f],
                'myopic' : [np.load('results/e2e/cal/myopic/'+f) for f in os.listdir('results/e2e/cal/myopic/') if f'{SHIFT}.npy' in f],
                'myopic_stupid' : [np.load('results/e2e/cal/myopic_stupid/'+f) for f in os.listdir('results/e2e/cal/myopic_stupid/') if f'{SHIFT}.npy' in f]
            },
            'opl' : {
                'relu' : {
                    'hist' : [np.load('results/e2e/opl_relu/historical/'+f) for f in os.listdir('results/e2e/opl_relu/historical/') if f'{SHIFT}.npy' in f],
                    'myopic' : [np.load('results/e2e/opl_relu/myopic/'+f) for f in os.listdir('results/e2e/opl_relu/myopic/') if f'{SHIFT}.npy' in f],
                    'myopic_stupid' : [np.load('results/e2e/opl_relu/myopic_stupid/'+f) for f in os.listdir('results/e2e/opl_relu/myopic_stupid/') if f'{SHIFT}.npy' in f]
                },
                'softplus' : {
                    'hist' : [np.load('results/e2e/opl_softplus//historical/'+f) for f in os.listdir('results/e2e/opl_softplus/historical/') if f'{SHIFT}.npy' in f],
                    'myopic' : [np.load('results/e2e/opl_softplus/myopic/'+f) for f in os.listdir('results/e2e/opl_softplus/myopic/') if f'{SHIFT}.npy' in f],
                    'myopic_stupid' : [np.load('results/e2e/opl_softplus/myopic_stupid/'+f) for f in os.listdir('results/e2e/opl_softplus/myopic_stupid/') if f'{SHIFT}.npy' in f]
                }
            }
        }

        mle = {
            'hist' : [np.load('results/mle/historical/'+f) for f in os.listdir('results/mle/historical/') if f'{SHIFT}.npy' in f],
            'myopic' : [np.load('results/mle/myopic/'+f) for f in os.listdir('results/mle/myopic/') if f'{SHIFT}.npy' in f],
            'myopic_stupid' : [np.load('results/mle/myopic_stupid/'+f) for f in os.listdir('results/mle/myopic_stupid/') if f'{SHIFT}.npy' in f]
        }

        for information in ['hist', 'myopic', 'myopic_stupid']:
            print(f'cal {information} '+str(np.vstack(e2e['cal'][information]).mean().round(3))+
                '('+str(np.std([np.mean(perf) for perf in e2e['cal'][information]]).round(3))+')')
            print(f'opl relu {information} '+str(np.vstack(e2e['opl']['relu'][information]).mean().round(3))+
                '('+str(np.std([np.mean(perf) for perf in e2e['opl']['relu'][information]]).round(3))+')')
            print(f'opl softplus {information} '+str(np.vstack(e2e['opl']['softplus'][information]).mean().round(3))
                +'('+str(np.std([np.mean(perf) for perf in e2e['opl']['softplus'][information]]).round(3))+')')
            print(f'mle {information} '+str(np.vstack(mle[information]).mean().round(3))+
                '('+str(np.std([np.mean(perf) for perf in mle[information]]).round(3))+')')