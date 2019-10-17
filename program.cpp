#include <iostream>
#include <cfloat>
#include <cmath>
#include <iomanip>

using namespace std;

int main(int argc, char const *argv[])
{
	int nstates;
	int nactions;
	int start_state, end_state;
	double discount_factor;
	string word;
	cin>>nstates;
	cin>>nactions;
	double reward[nstates][nactions][nstates];
	double transition[nstates][nactions][nstates];
	for (int i = 0; i < nstates; ++i)
	{
		for (int j = 0; j < nactions; ++j)
		{
			for (int k = 0; k < nstates; ++k)
			{
				cin>>reward[i][j][k];
			}
		}
	}
	for (int i = 0; i < nstates; ++i)
	{
		for (int j = 0; j < nactions; ++j)
		{
			for (int k = 0; k < nstates; ++k)
			{
				cin>>transition[i][j][k];
			}
		}
	}
	
	cin>>discount_factor;
	double * Vold;
	double * Vnew;
	int optimal_action[nstates];
	Vold = new double[nstates];
	Vnew = new double[nstates];
	for (int i = 0; i < nstates; ++i)
	{
		Vold[i]=0;
		Vnew[i]=0;
	}
	int iterating_variable = 0;
	while(true)
	{
		iterating_variable++;
		// cout<<"......\n";
		bool is_terminate = true;
		// cout<<"\nVold \n";
		// 	for (int i = 0; i < nstates; ++i)
		// 	{
		// 		cout<<Vold[i]<<endl;
		// 	}
		for (int i = 0; i < nstates; ++i)
		{

			double current_max_val = -DBL_MAX;
			int current_max_action = -1;
			double this_val;
			for(int j = 0 ; j<nactions ; j++)
			{	
				this_val=0;
				for (int k = 0; k < nstates; ++k)
				{
					this_val += transition[i][j][k]*(reward[i][j][k]+discount_factor*Vold[k]);
				}
				if(this_val>current_max_val)
				{
					// cout<<"Updating";
					current_max_val = this_val;
					current_max_action = j;
				}

			}
			Vnew[i] = current_max_val;
			if(abs(Vnew[i]-Vold[i])>=pow(10,-20))
				is_terminate = false;

			optimal_action[i] = current_max_action;
		}
		// cout<<"\nVold \t Vnew "<<endl;
		// for (int i = 0; i < nstates; ++i)
		// {
		// 	cout<<Vold[i]<<"\t"<<Vnew[i]<<endl;
		// }
		// delete Vold;
		for (int i = 0; i < nstates; ++i)
		{
			Vold[i] = Vnew[i];
		}
		
		
		if(is_terminate==true)
			break;
	}
	
	for (int i = 0; i < nstates; ++i)
	{
			cout<<double(Vnew[i])<<" "<<optimal_action[i]<<endl;
	}
	cout<<"iterations "<<iterating_variable<<endl;
	return 0;
}