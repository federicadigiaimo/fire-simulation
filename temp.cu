__global__ void fireKernel_Optimized(
    float* p_posX, float* p_posY,
    float* p_velX, float* p_velY,
    float* p_lifetime, unsigned int* p_rand_state,
    float* p_turbulence_flag,
    int numParticles,
    float dt,
    float time)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= numParticles) return;

    // --- SETUP SHARED MEMORY E CONDIZIONI ---
    extern __shared__ float s_data[];
    volatile float* s_turb_force_X = s_data;
    volatile float* s_turb_force_Y = s_data + 1;

    // Solo il primo thread del blocco determina se c'è un evento di turbolenza
    if (threadIdx.x == 0) {
        float turbulence_period = 2.5f; // Periodo più breve per vedere più spesso l'effetto
        float phase = fmodf(time, turbulence_period);
        // Attiva l'evento per una breve durata all'inizio del periodo
        if (phase < 0.1f) {
            unsigned int targetBlock = ((unsigned int)(time * 10.0f)) % gridDim.x;
            if (blockIdx.x == targetBlock) {
                float intensity = 10.0f;
                float angle = time * 2.0f;
                s_turb_force_X[0] = intensity * fast_cos(angle);
                s_turb_force_Y[0] = intensity * fast_sin(angle);
            }
            else {
                s_turb_force_X[0] = 0.0f;
                s_turb_force_Y[0] = 0.0f;
            }
        }
        else {
            s_turb_force_X[0] = 0.0f;
            s_turb_force_Y[0] = 0.0f;
        }
    }
    __syncthreads();

    // --- CARICAMENTO DATI NEI REGISTRI (UNA SOLA VOLTA) ---
    float l_posX = p_posX[gidx];
    float l_posY = p_posY[gidx];
    float l_velX = p_velX[gidx];
    float l_velY = p_velY[gidx];
    float l_lifetime = p_lifetime[gidx];
    unsigned int local_rand_state = p_rand_state[gidx];

    // Il flag di turbolenza decade lentamente, creando una "scia" di fumo
    float l_turbulence_flag = p_turbulence_flag[gidx] * 0.95f;

    // --- LOGICA PRINCIPALE ---
    if (l_lifetime <= 0.0f) {
        // RESPAWN
        l_posX = (random_float(local_rand_state) - 0.5f) * 0.2f;
        l_posY = -0.9f + (random_float(local_rand_state) * 0.15f);
        l_velX = (random_float(local_rand_state) - 0.5f) * 1.0f;
        l_velY = 2.0f + random_float(local_rand_state) * 2.0f;
        l_lifetime = 2.5f + random_float(local_rand_state) * 1.5f;
        l_turbulence_flag = 0.0f; // Il fumo non rinasce come fumo
    }
    else {
        // Controlla se il blocco è stato appena colpito dalla raffica di vento
        bool block_hit_by_turbulence = (s_turb_force_X[0] != 0.0f || s_turb_force_Y[0] != 0.0f);
        if (block_hit_by_turbulence) {
            // Se il blocco è colpito, tutte le particelle al suo interno diventano fumo
            l_turbulence_flag = 1.0f;
        }

        // Ora applichiamo la fisica in base allo stato (fuoco o fumo)
        if (l_turbulence_flag > 0.1f) {
            // --- FISICA DEL FUMO ---
            // Il fumo è più leggero e viene spinto dal "vento"
            l_velY += 0.5f * dt; // Sale più lentamente
            l_velX += s_turb_force_X[0] * dt; // Viene spinto dalla raffica
            l_velY += s_turb_force_Y[0] * dt;
            l_velX *= 0.99f; // Più resistenza dell'aria
            l_velY *= 0.99f;
        }
        else {
            // --- FISICA DEL FUOCO (la tua logica di prima) ---
            l_velX -= l_posX * 3.0f * dt;
            l_velY += 2.0f * dt;
            float turbulence = fast_sin(fmaf(l_posY, 3.0f, fmaf(time, 2.0f, l_posX * 2.0f)))
                + fast_cos(fmaf(l_posY, 5.0f, time * 2.5f));
            l_velX = fmaf(turbulence * 0.4f, dt, l_velX);
            float swirl = 0.3f * fast_sin(fmaf(time, 2.0f, l_posY * 4.0f));
            l_velX = fmaf(swirl, dt, l_velX);
            l_velX *= 0.985f;
            l_velY *= 0.992f;
        }

        // Aggiornamento comune
        l_posX += l_velX * dt;
        l_posY += l_velY * dt;
        l_lifetime -= dt;
    }

    // --- SCRITTURA FINALE (UNA SOLA VOLTA) ---
    p_posX[gidx] = l_posX;
    p_posY[gidx] = l_posY;
    p_velX[gidx] = l_velX;
    p_velY[gidx] = l_velY;
    p_lifetime[gidx] = l_lifetime;
    p_rand_state[gidx] = local_rand_state;
    p_turbulence_flag[gidx] = l_turbulence_flag;
}