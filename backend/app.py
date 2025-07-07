import os
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)

def classify_text_hf(text, token):
    url = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": f"Este é um email recebido por um assistente profissional. Classifique se é 'Produtivo' ou 'Improdutivo'. Conteúdo: {text}",
        "parameters": {
            "candidate_labels": ["Produtivo", "Improdutivo"]
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    label = result.get("labels", ["Improdutivo"])[0]
    return label

def generate_reply_hf(text, category, token):
    prompt = (
        f"Você é um assistente profissional. "
        f"Gere uma resposta breve e educada para um email classificado como '{category}'. "
        f"Email: {text}"
    )
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result = response.json()
        print("🔹 Prompt enviado:", prompt)
        print("🔹 Resposta da IA:", result)

        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            reply = result[0]["generated_text"]
        else:
            reply = (
                "Recebemos sua solicitação e encaminhamos para a equipe responsável. "
                "Agradecemos pelo contato e retornaremos em breve."
            )

    except Exception as e:
        print("⚠️ Erro na chamada à Hugging Face:", e)
        reply = (
            "Recebemos sua solicitação e encaminhamos para a equipe responsável. "
            "Agradecemos pelo contato e retornaremos em breve."
        )

    return reply.strip()

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(status="ok"), 200

@app.route("/classify", methods=["POST"])
def classify():
    text = ""

    if "email-file" in request.files and request.files["email-file"].filename:
        f = request.files["email-file"]
        name = f.filename.lower()
        data = f.read()

        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            for page in reader.pages:
                text += page.extract_text() or ""
        elif name.endswith(".txt"):
            # tenta múltiplas codificações comuns em várias regiões
            decodings = [
                # 🇧🇷 Brasil
                "utf-8", "utf-8-sig", "latin1", "cp1252", "ascii",
                # 🇺🇸 Estados Unidos
                "utf-16",
                # 🇪🇺 Europa
                "windows-1250",
                # 🌏 Ásia
                "shift_jis", "gbk", "euc-kr"
            ]
            for codec in decodings:
                try:
                    text = data.decode(codec)
                    print(f"🔹 Arquivo .txt decodificado com: {codec}")
                    break
                except Exception:
                    continue
            else:
                print("❌ Nenhuma codificação funcionou para o .txt")
                return jsonify(error="Não foi possível decodificar o arquivo .txt"), 400
        else:
            return jsonify(error="Formato não suportado"), 400
    else:
        text = request.form.get("email-text", "").strip()
        if not text:
            return jsonify(error="Nenhum conteúdo enviado"), 400

    hf_token = os.getenv("HF_API_TOKEN")
    texto_limpo = text.lower()

    # 🔍 Gatilhos de improdutividade: promocionais + feriados
    palavras_gatilho = [
        # Promocionais
        "promoção", "desconto", "grátis", "oferta", "imperdível", "cupom", "voucher", "brinde",
        "lançamento", "exclusivo", "últimas unidades", "compre agora", "aproveite", "liquidação",
        "preço especial", "preço baixo", "frete grátis", "amostra grátis", "black friday",
        "cyber monday", "aniversário de loja", "economize", "cashback", "vantagem", "novidade",
        "oportunidade", "ganhe", "garanta já", "por tempo limitado", "somente hoje", "último dia",
        "não perca", "corra", "cliente vip", "oferta relâmpago", "melhor preço",
        "condições especiais", "parcelamento facilitado", "sem juros",
        # Feriados e datas comemorativas
        "natal", "ano novo", "réveillon", "carnaval", "páscoa", "dia das mães", "dia dos pais",
        "dia dos namorados", "halloween", "feriado", "comemoração", "festivo", "celebração",
        "especial de natal", "oferta de páscoa", "promoção de feriado", "desconto de fim de ano",
        "liquidação de natal", "presentes", "presenteie", "ceia", "festa", "temporada de compras"
    ]

    frases_gatilho = [
        # Promocionais
        "essa oferta é para você", "não fique de fora", "você foi selecionado",
        "condição exclusiva para você", "promoção válida por tempo limitado",
        "aproveite enquanto dura", "só para clientes especiais", "temos uma surpresa para você",
        "você não pode perder", "olha essa novidade", "confira nossa nova coleção",
        "resgate seu cupom", "válido até hoje", "condições imperdíveis",
        "ideal para você economizar", "veja o que preparamos para você",
        "últimos dias da promoção", "essa é a sua chance", "oferta válida apenas hoje",
        "compre agora e economize", "promoção exclusiva online", "frete grátis em todo o site",
        "até 70% de desconto", "brinde especial para você", "ganhe mais por menos",
        "desconto especial para clientes fiéis", "clique e aproveite", "últimas unidades disponíveis",
        "condição nunca vista antes", "receba seu presente agora", "oportunidade única",
        "o melhor preço do mercado", "liquidação total", "não perca essa oportunidade",
        "exclusivo para assinantes", "parcelamento em até 12x sem juros", "só até amanhã",
        "acelere e aproveite", "leve 3 e pague 2", "compre um e leve outro",
        # Feriados e datas comemorativas
        "especial de natal", "celebre o natal com a gente", "comemore o ano novo em grande estilo",
        "promoção de fim de ano", "descontos de natal imperdíveis", "presentes para todos os gostos",
        "ofertas natalinas", "liquidação de ano novo", "boas festas com economia",
        "esquente seu carnaval com ofertas", "promoção de páscoa", "celebre com descontos especiais",
        "leve o presente ideal", "promoção para o dia das mães", "presentes para o dia dos pais",
        "amor e ofertas no ar", "descontos apaixonantes", "ofertas assustadoras de halloween",
        "só hoje: oferta de páscoa", "venha conferir nossas ofertas natalinas",
        "natal premiado para você", "entre no clima com nossas ofertas",
        "o presente perfeito está aqui", "tempo de economizar",
        "presentes inesquecíveis com desconto", "liquidação pós-feriado",
        "comemore economizando", "mais alegria, menos preço",
        "promoções temáticas incríveis", "boas festas e bons preços"
    ]

    # 🔎 Gatilhos de produtividade técnica
    gatilhos = [
        "erro", "falha", "urgente", "problema", "suporte", "ajuda",
        "travando", "bug", "inacessível", "crítico", "instabilidade",
        "parou", "lentidão", "inconsistência", "não funciona", "não carrega",
        "não consigo acessar", "não abre", "não entra", "não responde",
        "não envia", "não reconhece", "sistema caiu", "fora do ar", "apagou tudo",
        "me desconectou", "dados sumiram", "login inválido", "tela branca",
        "tela preta", "formulário travado", "formulário com erro", "crash",
        "não consigo concluir", "não consigo finalizar", "não salva", "código de erro",
        "erro 500", "erro 404", "erro interno", "não consigo emitir boleto",
        "erro na nota fiscal", "falha no pagamento", "problema financeiro",
        "não gerou fatura", "estou sem faturamento", "cliente não recebeu",
        "venda não concluída", "pedido não foi processado", "impacta minha operação",
        "interrompeu minhas vendas", "me gerou custo", "vou ter que parar tudo",
        "estou sendo cobrado", "isso afeta a entrega", "impacta contrato",
        "estou atrasado por causa disso", "isso pode gerar multa", "questão legal",
        "problema jurídico", "abri chamado", "ticket", "aguardo contato",
        "ninguém me respondeu", "não tive retorno", "péssima experiência",
        "muito ruim", "não estou satisfeito", "inaceitável", "decepcionante",
        "insuportável", "vou cancelar", "nunca mais uso", "falta de respeito",
        "esperando há dias", "ninguém resolve", "já tentei de tudo",
        "problema recorrente", "isso acontece sempre", "já tive esse erro antes",
        "quero falar com alguém", "como posso resolver", "passo a passo",
        "preciso falar com alguém", "me ajudem", "socorro"
    ]

    # 🧠 Lógica de classificação
    if any(frase in texto_limpo for frase in frases_gatilho) or any(palavra in texto_limpo for palavra in palavras_gatilho):
        category = "Improdutivo"
    elif any(g in texto_limpo for g in gatilhos):
        category = "Produtivo"
    else:
        category = classify_text_hf(text, hf_token)

    if category == "Improdutivo":
        reply = "Este email não requer resposta automática."
    else:
        reply = generate_reply_hf(text, category, hf_token)

    return jsonify(category=category, suggestion=reply), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)