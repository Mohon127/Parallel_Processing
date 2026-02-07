#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// Function to send a large string over MPI
void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

// Function to receive a large string over MPI
string receive_string(int sender) {
    int len;
    MPI_Status status;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, &status);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, &status);
    string res(buf);
    delete[] buf;
    return res;
}

// Converts a range of a vector of strings into one single string for transmission
string vector_to_string(const vector<string> &lines, int start, int end) {
    string result;
    for (int i = start; i < min((int)lines.size(), end); i++) {
        result += lines[i] + "\n";
    }
    return result;
}

// Splits a large received string back into a vector of strings
vector<string> string_to_vector(const string &text) {
    vector<string> lines;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

// Reads raw lines from multiple files into a vector
void read_phonebook(const vector<string> &files, vector<string> &lines) {
    for (const string &file : files) {
        ifstream f(file);
        if (!f.is_open()) {
            cerr << "Could not open file: " << file << endl;
            continue;
        }
        string line;
        while (getline(f, line)) {
            if (!line.empty()) lines.push_back(line);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file1>... <search_term>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start_time, end_time;

    if (rank == 0) {
        // --- MASTER PROCESS ---
        vector<string> files;
        for (int i = 1; i < argc - 1; i++) files.push_back(argv[i]);

        vector<string> all_lines;
        read_phonebook(files, all_lines);

        int total = all_lines.size();
        int chunk = (total + size - 1) / size;

        // Distribute data to workers
        for (int i = 1; i < size; i++) {
            string text_chunk = vector_to_string(all_lines, i * chunk, (i + 1) * chunk);
            send_string(text_chunk, i);
        }

        // Start global timer
        start_time = MPI_Wtime();

        // --- MASTER CHUNK SEARCH ---
        double master_start = MPI_Wtime();
        vector<string> final_matches;
        for (int i = 0; i < min(chunk, total); i++) {
            if (all_lines[i].find(search_term) != string::npos) {
                final_matches.push_back(all_lines[i]);
            }
        }
        double master_end = MPI_Wtime();
        // printf("Master process searched %d lines in %f seconds.\n",
        //        min(chunk, total), master_end - master_start);
        // print time to process master chunk
        printf("Process %d in time %f seconds.\n", rank, master_end - master_start);

        // Receive results from workers
        for (int i = 1; i < size; i++) {
            string worker_raw_res = receive_string(i);
            vector<string> worker_vec = string_to_vector(worker_raw_res);
            final_matches.insert(final_matches.end(), worker_vec.begin(), worker_vec.end());
        }

        // Sort results
        sort(final_matches.begin(), final_matches.end());

        end_time = MPI_Wtime();

        // Write results
        ofstream out("output.txt");
        for (const string &match : final_matches) {
            out << match << "\n";
        }
        out.close();

        cout << "Search complete. Found " << final_matches.size() << " matches." << endl;
        printf("Total execution time : %f seconds.\n",
               end_time - start_time);

    } else {
        // --- WORKER PROCESS ---
        string recv_text = receive_string(0);
        vector<string> local_lines = string_to_vector(recv_text);
        
        double worker_start = MPI_Wtime();
        string local_matches_str = "";
        for (const string &line : local_lines) {
            if (line.find(search_term) != string::npos) {
                local_matches_str += line + "\n";
            }
        }
        double worker_end = MPI_Wtime();

        // Send local results back to Master
        send_string(local_matches_str, 0);
        // printf("Process %d processed %lu lines in %f seconds.\n",
        //        rank, local_lines.size(), worker_end - worker_start);
        printf("Process %d in time %f seconds.\n", rank, worker_end - worker_start);
    }

    MPI_Finalize();
    return 0;
}


/*
mpic++ phone_book.cpp -o phone_book
mpirun -n 4 ./phone_book input.txt 'TUMPA'
*/