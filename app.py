from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
import scipy.stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.config['SESSION_COOKIE_SIZE_LIMIT'] = 50000 #had troble with bigger samples until I see piazza
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S): # to do complete
    # Generate data and initial plots
    print("\n=== Starting generate_data ===")
    print(f"Input parameters: N={N}, mu={mu}, beta0={beta0}, beta1={beta1}, sigma2={sigma2}, S={S}")
  
    
    X =  np.random.uniform(0, 1, N)  # Replace with code to generate random values for X
    X = X.reshape(-1,1)
    
    # Y = beta0 + beta1 * X + mu + error term
    # we used this equation in lab too
    error = np.random.normal(mu, np.sqrt(sigma2), N).reshape(-1, 1)
    true_relationship = beta0 + beta1 * X  
    Y = true_relationship + error # Replace with code to generate Y

    model =  LinearRegression()  # Initialize the LinearRegression model
    model.fit(X, Y)
    # Fit the model to X and Y
    slope = model.coef_[0][0]  # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_[0]  # Extract the intercept from the fitted model

    plt.figure(figsize=(12, 8))
    plt.scatter(X, Y, alpha=0.5, color='blue', label='Data point')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.plot(X, true_relationship, color='green', linewidth=2, linestyle='--', 
            label=f'True relationship (Y = {beta0:.2f} + {beta1:.2f}X)')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Regression: Y = {slope:.2f}X + {intercept:.2f}")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    # Replace with code to generate and save the scatter plot
    
    # ----------------- Debug Statements ------------------
    print(f"\nInitial fit results:") 
    print(f"Fitted slope: {slope:.3f}")
    print(f"True slope (beta1): {beta1}")
    print(f"Fitted intercept: {intercept:.3f}")
    print(f"True intercept (beta0): {beta0}")
    
    print("\nStarting simulations...")
    # ----------------- Debug Statements ------------------


    slopes = []
    intercepts = []

    for _ in range(S):

        X_sim = np.random.uniform(0, 1, N)  # Replace with code to generate simulated X values
        # from last week
        X_sim = X_sim.reshape(-1, 1) 
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N).reshape(-1, 1)
        Y_sim = beta0 + beta1 * X_sim + error_sim  # Replace with code to generate simulated Y values

        sim_model =  LinearRegression()  # Replace with code to fit the model
        sim_model.fit(X_sim, Y_sim)
        sim_slope = sim_model.coef_[0][0]  
        sim_intercept = sim_model.intercept_[0] 

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, 
                label=f"Observed Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, 
                label=f"Observed Intercept: {intercept:.2f}")
    plt.axvline(beta1, color="green", linestyle="--", linewidth=1, 
                label=f"True Slope: {beta1:.2f}")
    plt.axvline(beta0, color="lime", linestyle="--", linewidth=1, 
                label=f"True Intercept: {beta0:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    # Replace with code to generate and save the histogram plot


    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(abs(s - beta1) > abs(slope - beta1) for s in slopes) / S  # Replace with code to calculate proportion of slopes more extreme than observed
    intercept_extreme = sum(abs(i - beta0) > abs(intercept - beta0) for i in intercepts) / S  # Replace with code to calculate proportion of intercepts more extreme than observed

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])
        print(f"Form parameters: N={N}, mu={mu}, sigma2={sigma2}, beta0={beta0}, beta1={beta1}, S={S}")
        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        print("\nStoring in session...")
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # ----------------- Debug Statements ------------------
        print("Verification of session storage:")
        print(f"Stored slope: {session['slope']}")
        print(f"Stored beta1: {session['beta1']}")
        print(f"First few stored slopes: {session['slopes'][:5]}")

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    try:
       
        
        slopes = np.array(session.get("slopes", []))
        beta1 = float(session.get("beta1", 0))
        slope = float(session.get("slope", 0))
        N = int(session.get("N", 0))
        S = int(session.get("S", 0))

        # ----------------- Debug Statements ------------------
        print(f"Slopes from session: {slopes[:5]}")
        print(f"Beta1 from session: {beta1}")
        print(f"Observed slope: {slope}")

        parameter = request.form.get("parameter")
        test_type = request.form.get("test_type")

        # Use the slopes or intercepts from the simulations
        simulated_stats = slopes
        observed_stat = slope
        hypothesized_value = beta1

        #the html has the key of < > !=
        if test_type == "!=":
            p_value = np.mean(abs(simulated_stats - hypothesized_value) >= 
                            abs(observed_stat - hypothesized_value))
        elif test_type == ">":
            p_value = np.mean(simulated_stats >= observed_stat)
        else:  # test_type == "<"
            p_value = np.mean(simulated_stats <= observed_stat)

        if p_value <= 0.0001:
            fun_message = "* Wow! This is extremely statistically significant!"
        elif p_value <= 0.01:
            fun_message = "** Very strong evidence against the null hypothesis!"
        elif p_value <= 0.05:
            fun_message = "*** We've found statistical significance!"
        else:
            fun_message = "! Not quite significant at conventional levels. Keep exploring!"


        plt.figure(figsize=(12, 6))
        # Create histogram and get the bin edges
        counts, bins, _ = plt.hist(simulated_stats, bins=30, density=True, alpha=0.6, 
                                 color='skyblue', label='Simulated values')
        bin_width = bins[1] - bins[0]
        
        # Add vertical lines for observed and hypothesized values
        plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2,
                   label=f'Observed value: {observed_stat:.3f}')
        plt.axvline(hypothesized_value, color='green', linestyle='--', linewidth=2,
                   label=f'Hypothesized value: {hypothesized_value:.3f}')

        # Create x values for fill_between 
        x = bins[:-1] + bin_width/2
        y = counts

        # to do for visual
        if test_type == "!=": # shading
            diff = abs(observed_stat - hypothesized_value)
            mask = abs(x - hypothesized_value) >= diff
            plt.fill_between(x[mask], 0, y[mask], color='red', alpha=0.2)
        elif test_type == ">":
            mask = x >= hypothesized_value
            plt.fill_between(x[mask], 0, y[mask], color='red', alpha=0.2)
        else:  # test_type == "<"
            mask = x <= hypothesized_value
            plt.fill_between(x[mask], 0, y[mask], color='red', alpha=0.2)

        plt.title(f'Hypothesis Test for {parameter.title()}\nP-value: {p_value:.4f}')
        plt.xlabel(f'{parameter.title()} Values')
        plt.ylabel('Density')
        plt.legend()

        plot3_path = "static/plot3.png"
        plt.savefig(plot3_path, bbox_inches='tight')
        plt.close()

        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot3=plot3_path,
            parameter=parameter,
            observed_stat=observed_stat,
            hypothesized_value=hypothesized_value,
            N=N,
            beta0=session.get("beta0"),
            beta1=beta1,
            S=S,
            p_value=p_value,
            fun_message=fun_message,
        )

    except Exception as e: # debugging 
        print(f"Error in hypothesis_test: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", error=str(e))

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        N = int(session.get("N"))
        mu = float(session.get("mu"))
        sigma2 = float(session.get("sigma2"))
        beta0 = float(session.get("beta0"))
        beta1 = float(session.get("beta1"))
        S = int(session.get("S"))
        slope = float(session.get("slope"))
        intercept = float(session.get("intercept"))
        parameter = request.form.get("parameter", "slope")
        
        
        confidence_level_str = request.form.get("confidence_level", "95")
        confidence_level = float(confidence_level_str) / 100  # convert percentage to decimal
        
        # ---------- Debugging -------------
        print(f"Raw confidence level string: {confidence_level_str}")
        print(f"Converted confidence level: {confidence_level}")

        # Get the appropriate estimates and parameters
        if parameter == "slope":
            estimates = np.array(session.get("slopes"))
            observed_stat = slope
            true_param = beta1
        else:
            estimates = np.array(session.get("intercepts"))
            observed_stat = intercept
            true_param = beta0

        # ---------- Debugging -------------
        print(f"Number of estimates: {len(estimates)}")
        print(f"Estimates range: [{min(estimates):.4f}, {max(estimates):.4f}]")
        
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates, ddof=1)
        print(f"Mean estimate: {mean_estimate:.4f}")
        print(f"Standard deviation: {std_estimate:.4f}")
        
      
     
        alpha = 1 - confidence_level # calculate alpha 
        df = len(estimates) - 1
        
        # ---------- Debugging -------------
        print(f"Confidence level: {confidence_level}")
        print(f"Alpha: {alpha:.4f}")
        print(f"Degrees of freedom: {df}")
        
        t_value = scipy.stats.t.ppf(1 - alpha/2, df=df)
        print(f"T-value: {t_value:.4f}")
        
        std_error = std_estimate / np.sqrt(len(estimates))
        print(f"Standard error: {std_error:.4f}")
        
        margin_of_error = t_value * std_error
        print(f"Margin of error: {margin_of_error:.4f}")
        
        ci_lower = mean_estimate - margin_of_error
        ci_upper = mean_estimate + margin_of_error
        
        print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        includes_true = (ci_lower <= true_param <= ci_upper)
        print(f"Includes true parameter: {includes_true}")
            
        plt.figure(figsize=(10, 6))
        y_value = 0
        
        plt.scatter(estimates, [y_value] * len(estimates), 
                   color='gray', alpha=0.6, s=30,
                   label='Simulated Estimates')
        
        # mean estimate
        plt.scatter([mean_estimate], [y_value], color='blue', s=100, 
                   label='Mean Estimate')
        # confidence interval
        plt.hlines(y=y_value, xmin=ci_lower, xmax=ci_upper, 
                  colors='blue', linewidth=2,
                  label=f'{confidence_level*100:.1f}% Confidence Interval')
        #true param? 
        plt.axvline(x=true_param, color='green', linestyle='--',
                   label='True Slope')
        
        # Formatting
        plt.title(f'{confidence_level*100:.1f}% Confidence Interval for Slope (Mean Estimate)',
                 pad=20)
        plt.xlabel('Slope Estimate')
        
        # remove y-axis ticks and labels
        plt.yticks([])
        
        # Set x-axis limits with proper padding
        x_range = max(estimates) - min(estimates)
        x_min = min(estimates) - 0.1 * x_range
        x_max = max(estimates) + 0.1 * x_range
        plt.xlim(x_min, x_max)
        
        plt.ylim(-0.5, 0.5)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.1, axis='x')
        
        plot4_path = "static/plot4.png"
        plt.savefig(plot4_path, bbox_inches='tight', dpi=300)


        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot4=plot4_path,
            parameter=parameter,
            confidence_level=float(confidence_level_str), 
            mean_estimate=mean_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            includes_true=includes_true,
            observed_stat=observed_stat,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )

    except Exception as e:
        print(f"\n=== Error in confidence_interval ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port = 3000)
