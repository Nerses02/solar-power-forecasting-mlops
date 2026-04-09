import subprocess
import sys
import time

def run_script(script_name):
    """
    Այս ֆունկցիան աշխատացնում է նշված Python ֆայլը և ստուգում է արդյունքը:
    """
    print(f"\n{'='*50}")
    print(f"🚀 ՍԿՍՎՈՒՄ Է: {script_name}")
    print(f"{'='*50}\n")
    
    try:
        # sys.executable-ը ավտոմատ գտնում է քո համակարգչի ճիշտ python/py հրամանը
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ ՀԱՋՈՂՈՒԹՅԱՄԲ ԱՎԱՐՏՎԵՑ: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ՍԽԱԼ ՏԵՂԻ ՈՒՆԵՑԱՎ '{script_name}'-ի աշխատանքի ժամանակ:")
        print("Խողովակաշարի աշխատանքը դադարեցված է:")
        sys.exit(1) # Կանգնեցնում ենք ծրագիրը, որ հաջորդ քայլերին չանցնի

def main():
    print("🌟 ԱՐԵՎԱՅԻՆ ԷՆԵՐԳԻԱՅԻ ԿԱՆԽԱՏԵՍՄԱՆ ԱՎՏՈՄԱՏԱՑՎԱԾ ՀԱՄԱԿԱՐԳ 🌟")
    print("Ամբողջական խողովակաշարը (Pipeline) սկսում է իր աշխատանքը...\n")
    
    start_time = time.time()
    
    # Այստեղ նշում ենք այն հաջորդականությունը, որով պետք է աշխատեն սկրիպտները
    pipeline_scripts = [
        "1_fetch_nasa_history.py",
        "2_merge_datasets.py",
        "3_train_models.py",
        "4_fetch_future_weather.py",
        "5_predict_and_plot.py"
    ]
    
    # Ցիկլով հերթով կանչում ենք բոլորին
    for script in pipeline_scripts:
        run_script(script)
        # Մի փոքր դադար ենք տալիս սկրիպտների միջև՝ համակարգը չծանրաբեռնելու համար
        time.sleep(1) 
        
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    print(f"\n{'*'*50}")
    print(f"🎉 ՇՆՈՐՀԱՎՈՐՈՒՄ ԵՄ: ԱՄԲՈՂՋ ՀԱՄԱԿԱՐԳՆ ԱՎԱՐՏԵՑ ԻՐ ԱՇԽԱՏԱՆՔԸ!")
    print(f"⏱ Ընդհանուր պահանջված ժամանակը: {total_time:.1f} րոպե:")
    print(f"{'*'*50}")

if __name__ == "__main__":
    main()