#include <iostream>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <map>
#include <tuple>
#include <vector>

using namespace std;

int main(int argc, char const *argv[])
{
	int nstates;
	int nactions;
	int start_state, end_state;
	double discount_factor;
	string word;
	cin>>word;
	if(word=="numStates")
		cin>>nstates;
	cin>>word;
	if(word=="numActions")
		cin>>nactions;
	cin>>word;
	if(word=="start")
		cin>>start_state;
	cin>>word;
	if(word=="end")
		cin>>end_state;
	map<tuple<int,int> , vector<tuple<int,double> > > rewardmap;
	map<tuple<int,int> , vector<tuple<int, double> > > transitionmap;
	// double reward[nstates][nactions][nstates];
	// double transition[nstates][nactions][nstates];

	for (int i = 0; i < nstates; ++i)
	{
		for (int j = 0; j < nactions; ++j)
		{
			vector<tuple<int, double> > emptyvector;
			tuple<int, int> curtuple(i,j);
			rewardmap[curtuple] = emptyvector;
			transitionmap[curtuple] = emptyvector;
			// for (int k = 0; k < nstates; ++k)
			// {
			// 	transition[i][j][k]=0;
			// 	reward[i][j][k] = 0;
			// }
		}
	}
	cin>>word;
	while(word=="transitions")
	{
		int s1,s2,ac;
		double r,p;
		cin>>s1>>ac>>s2>>r>>p;
		// transition[s1][ac][s2] = p;
		// reward[s1][ac][s2] = r;

		tuple<int, int> curtuple(s1,ac);
		tuple<int, double> Ttuple(s2,p);
		tuple<int, double> Rtuple(s2,r);
		rewardmap[curtuple].push_back(Rtuple);
		transitionmap[curtuple].push_back(Ttuple);
		cin>>word;
		// cout<<word<<(word=="transitions");

	}

	if(word=="discount")
	{
		// cout<<"ASfasdfsdfgsdg";
		cin>>discount_factor;
	}
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
		
		for (int i = 0; i < nstates; ++i)
		{
			if(i==end_state)
			{
				Vnew[i]=0;
				continue;
			}

			double current_max_val = -DBL_MAX;
			int current_max_action = -1;
			double this_val;
			for(int j = 0 ; j<nactions ; j++)
			{	
				this_val=0;
				int check=0;
				tuple<int, int> curtuple(i,j);
				// for (int k = 0; k < nstates; ++k)
				// {
				// 	this_val += transition[i][j][k]*(reward[i][j][k]+discount_factor*Vold[k]);
				// }
				for (int k = 0; k < transitionmap[curtuple].size(); ++k)
				{
					check=1;
					this_val += get<1>(transitionmap[curtuple][k])*(get<1>(rewardmap[curtuple][k]) + discount_factor*Vold[get<0>(rewardmap[curtuple][k])]);
				}
				if(this_val>current_max_val && check==1)
				{
					// cout<<"Updating";
					current_max_val = this_val;
					current_max_action = j;
				}

			}
			Vnew[i] = current_max_val;
			if(abs(Vnew[i]-Vold[i])>=pow(10,-16))
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
		if(i==end_state)
			cout<<double(0)<<" "<<-1<<endl;
		else
			cout<<double(Vnew[i])<<" "<<optimal_action[i]<<endl;
	}
	cout<<"iterations "<<iterating_variable<<endl;
	return 0;
}